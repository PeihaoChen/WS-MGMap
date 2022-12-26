import numpy as np
from gym import Space

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.encoders.instruction_encoder import InstructionEncoder
from vlnce_baselines.models.encoders.unet_encoder import UNet
from vlnce_baselines.models.encoders.resnet_encoders import VlnResnetDepthEncoder
from vlnce_baselines.models.encoders.map_encoder import MapEncoder, MapDecoder
from vlnce_baselines.common.rgb_mapping import RGBMapping


class MGMapNet(Net):
    """ A multi-granularity map (MGMap) network that contains:
        Instruction encoder
        RGB encoder
        Depth encoder
        Map encoder and decoder
        RNN state encoder
    """
    def __init__(self, observation_space: Space, model_config: Config):
        super().__init__()
        self.model_config = model_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)

        # # Init the rgb encoder
        self.rgb_encoder = UNet(model_config)
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the depth encoder
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=True,
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the mapping network 
        self.rgb_mapping_module = RGBMapping(model_config.RGBMAPPING)
        map_channel = model_config.RGBMAPPING.map_depth

        # Init the map encoder
        self.map_encoder = MapEncoder(
            model_config.MAP_ENCODER.ego_map_size,
            map_channel,
            model_config.MAP_ENCODER.output_size,
        )

        # Init the map decoder
        self.map_decoder = MapDecoder(model_config.MAP_ENCODER.output_size)
        self.map_classfier = nn.Sequential(
            nn.ConvTranspose2d(self.map_decoder.output_shape[0], 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 27, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # Init the map linear
        self.map_encoded_linear = nn.Sequential(
            nn.Conv2d(self.map_encoder.output_shape[0], 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.map_classified_linear = nn.Sequential(
            nn.Conv2d(27, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.map_cated_linear = nn.Sequential(
            nn.Conv2d(128*2, model_config.MAP_ENCODER.output_size, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.map_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                model_config.MAP_ENCODER.output_size,
                model_config.MAP_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the first rnn state decoder
        self._hidden_size = model_config.STATE_ENCODER.hidden_size
        first_state_input_size = (
            (model_config.RGB_ENCODER.output_size if 'rgb' in model_config.STATE_ENCODER.input_type else 0)
            + (model_config.DEPTH_ENCODER.output_size if 'depth' in model_config.STATE_ENCODER.input_type else 0)
            + (model_config.MAP_ENCODER.output_size if 'map' in model_config.STATE_ENCODER.input_type else 0)
        )
        self.state_encoder = RNNStateEncoder(
            input_size=first_state_input_size,
            hidden_size=self._hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        # Init the attention encoder
        self.state_text_q_layer = nn.Linear(self._hidden_size, self._hidden_size // 2)
        self.state_text_k_layer = nn.Conv1d(self.instruction_encoder.output_size, self._hidden_size // 2, 1)

        self.text_map_q_layer = nn.Linear(self.instruction_encoder.output_size, self._hidden_size // 2)
        self.text_map_k_layer = nn.Conv1d(self.map_encoder.output_shape[0], self._hidden_size // 2, 1)
        
        self.register_buffer("_scale", torch.tensor(1.0 / ((self._hidden_size // 2) ** 0.5)))

        # Init the second rnn state decoder
        second_state_input_size = (
            model_config.STATE_ENCODER.hidden_size
            + model_config.STATE_ENCODER.hidden_size // 2
            + (model_config.STATE_ENCODER.hidden_size // 2 if 'map' in model_config.STATE_ENCODER.input_type else 0)
        )
        self.second_state_compress = nn.Sequential(
            nn.Linear(
                second_state_input_size,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )
        self.second_state_encoder = RNNStateEncoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )
        self._output_size = model_config.STATE_ENCODER.hidden_size

        self.train()
        self.depth_encoder.eval()
        self.rgb_encoder.eval()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)
        if mask is not None:
            logits = logits - mask.float() * 1e8
        attn = F.softmax(logits * self._scale, dim=1)
        return torch.einsum("ni, nci -> nc", attn, v), attn

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        instruction_embedding, text_mask = self.instruction_encoder(observations)
        rgb_embedding, rgb_embedding_proj = self.rgb_encoder(observations)
        depth_embedding = self.depth_encoder(observations)

        # Get map
        self.rgb_mapping_module(rgb_embedding_proj, observations, masks)
        ego_map = observations['rgb_ego_map']

        # Encoding map
        map_encoded = self.map_encoder(ego_map)
        map_encoded_proj = self.map_encoded_linear(map_encoded)

        # Decoding map (ie segmentation prediction)
        map_decoded = self.map_decoder(map_encoded)   # [bs, 64, 64]
        pred_sem_map = self.map_classfier(map_decoded) 
        map_classified_proj = self.map_classified_linear(
            torch.nn.functional.avg_pool2d(pred_sem_map, kernel_size=2, stride=2)
        )

        # Get concated map embedding
        map_cat = [map_encoded_proj, map_classified_proj]
        map_embedding = torch.cat(map_cat, dim=1)   # [bs, 2*c / c, 50, 50]
        map_embedding = self.map_cated_linear(map_embedding)

        rgb_embedding = torch.flatten(rgb_embedding, 2)
        depth_embedding = torch.flatten(depth_embedding, 2)
        map_embedding = torch.flatten(map_embedding, 2)

        state_in = []
        if 'rgb' in self.model_config.STATE_ENCODER.input_type:
            rgb_in = self.rgb_linear(rgb_embedding)
            state_in.append(rgb_in)
        if 'depth' in self.model_config.STATE_ENCODER.input_type:
            depth_in = self.depth_linear(depth_embedding)
            state_in.append(depth_in)
        if 'map' in self.model_config.STATE_ENCODER.input_type:
            map_in = self.map_linear(map_embedding)
            state_in.append(map_in)
        state_in = torch.cat(state_in, dim=1)
        (
            state,
            rnn_hidden_states[0: self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_hidden_states[0: self.state_encoder.num_recurrent_layers],
            masks,
        )

        state_text_q = self.state_text_q_layer(state)
        state_text_k = self.state_text_k_layer(instruction_embedding)
        text_embedding, _ = self._attn(state_text_q, state_text_k, instruction_embedding, text_mask)

        text_map_q = self.text_map_q_layer(text_embedding)
        text_map_k = self.text_map_k_layer(map_embedding)
        map_embedding, self.att_map_t_m = self._attn(text_map_q, text_map_k, map_embedding, None)

        if 'map' in self.model_config.STATE_ENCODER.input_type:
            x = torch.cat([state, text_embedding, map_embedding], dim=1)
        else:
            x = torch.cat([state, text_embedding], dim=1)
        x = self.second_state_compress(x)
        (
            x,
            rnn_hidden_states[self.state_encoder.num_recurrent_layers:],
        ) = self.second_state_encoder(
            x,
            rnn_hidden_states[self.state_encoder.num_recurrent_layers:],
            masks
        )

        return x, rnn_hidden_states, pred_sem_map
