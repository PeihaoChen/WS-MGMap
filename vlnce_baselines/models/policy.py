from gym import Space

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat import Config
from habitat_baselines.rl.ppo.policy import CriticHead

from vlnce_baselines.models.mg_map_policy import MGMapNet
from vlnce_baselines.common.distributions import DiagGaussian
from vlnce_baselines.common.aux_losses import AuxLosses


class BasePolicy(nn.Module):
    def __init__(self, observation_space: Space, action_space: Space, model_config: Config):
        super(BasePolicy, self).__init__()
        self.model_config = model_config

        # Forward Network
        self.net = MGMapNet(observation_space, model_config)

        # Actor_critic Network
        self.action_distribution = DiagGaussian(self.net.output_size, action_space.shape[0])
        self.critic = CriticHead(self.net.output_size)

        # Aux Network
        self.prog_pred = nn.Linear(model_config.STATE_ENCODER.hidden_size, 1)

    def update_map(self, observations, masks):
        _, rgb_embedding_proj = self.net.rgb_encoder(observations)
        self.net.rgb_mapping_module(rgb_embedding_proj, observations, masks)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, pred_map = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        self.aux_prediction(features, observations, pred_map)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def aux_prediction(self, features, observations, pred_map):
        self.prog = torch.tanh(self.prog_pred(features))

        # Calculate loss
        if AuxLosses.is_active():
            if self.model_config.PREDICTION_MONITOR.use:
                target_map = torch.nn.functional.interpolate(observations['gt_semantic_map'].unsqueeze(1), size=(48, 48)).squeeze().long()
                prediction_loss = F.cross_entropy(pred_map, target_map, reduction='none')
                prediction_loss = prediction_loss.mean([1,2])
                AuxLosses.register_loss('prediction_monitor', prediction_loss, self.model_config.PREDICTION_MONITOR.alpha)
            
            if self.model_config.CONTRASTIVE_MONITOR.use:
                feature_size = self.net.map_encoder.output_shape[-1]

                if 'gt_path' in observations.keys():
                    dis_map = observations['gt_path']
                else:
                    dis_map = observations['waypoint_distribution']
                target = (dis_map.max() - dis_map) / (dis_map.max() - dis_map.min())
                target = F.interpolate(target.unsqueeze(1), size=[feature_size, feature_size], mode='area').squeeze(1)
                target = target.reshape(target.shape[0], -1)
                target = F.softmax(target/self.model_config.CONTRASTIVE_MONITOR.target_tau, dim=1)
                pred = self.net.att_map_t_m
                
                kl_loss = F.kl_div(torch.log(pred), target, reduction='none')
                kl_loss = kl_loss.mean(-1)
                AuxLosses.register_loss('contrastive_monitor', kl_loss, self.model_config.CONTRASTIVE_MONITOR.alpha)

            if self.model_config.PROGRESS_MONITOR.use:
                progress_loss = F.mse_loss(self.prog, observations['progress'], reduction='none')
                progress_loss = progress_loss.mean(-1)
                AuxLosses.register_loss('progress_monitor', progress_loss, self.model_config.PROGRESS_MONITOR.alpha)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, weights):
        features, rnn_hidden_states, pred_map = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        pred = distribution.mean

        self.aux_prediction(features, observations, pred_map)
        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)

        return pred, aux_loss
