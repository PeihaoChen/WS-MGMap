import gc
import json
import os
import time
import tqdm
import math
import datetime
import numpy as np
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs, generate_video, poll_checkpoint_folder
from habitat_extensions.utils import observations_to_image

from vlnce_baselines.models.policy import BasePolicy
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import transform_obs


class CommonTrainer(BaseRLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.envs = None

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=18000))
        self.device = (
            torch.device("cuda", self.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"[init] == local rank: {self.local_rank}")

    def _setup_actor_critic(
        self, config: Config, load_from_ckpt: bool, ckpt_path: str
    ) -> None:
        """Sets up actor critic and agent.
        Args:
            config: config
        Returns:
            None
        """
        self.actor_critic = BasePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            model_config=config.MODEL,
        )
        self.actor_critic.to(self.device)
        self.actor_critic = DDP(
            self.actor_critic,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True
        )
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=self.config.DAGGER.LR
        )

        if load_from_ckpt:
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            ckpt_dict["state_dict"] = {'module.'+k: v for k, v in ckpt_dict["state_dict"].items()}
            msg = self.actor_critic.load_state_dict(ckpt_dict["state_dict"], strict=False)
            logger.warning(f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")
        logger.info("Finished setting up actor critic model.")

        if self.local_rank == 0:
            logger.info(
                "agent number of parameters: {}".format(
                    sum(param.numel() for param in self.actor_critic.parameters())
                )
            )
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(p.numel() for p in self.actor_critic.parameters() if p.requires_grad)
                )
            )

    def save_checkpoint(self, file_name, extra_state: Optional[Dict] = None) -> None:
        """Save checkpoint with specified name.
        Args:
            file_name: file name for checkpoint
        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.actor_critic.module.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        """Load checkpoint of specified path as a dict.
        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args
        Returns:
            dict containing checkpoint info
        """
        ckpt = torch.load(checkpoint_path, *args, **kwargs)
        return ckpt

    def resume_dagger(self):
        start_dagger_it = 0
        start_epoch_it = 0

        ckpt_file = None
        if self.config.RESUME_CKPT is not None:
            ckpt_file = self.config.RESUME_CKPT
        if len(os.listdir(self.config.CHECKPOINT_FOLDER)) != 0:
            dir_list = sorted(os.listdir(self.config.CHECKPOINT_FOLDER), key=lambda x: os.path.getmtime(os.path.join(self.config.CHECKPOINT_FOLDER, x)))
            ckpt_file = os.path.join(self.config.CHECKPOINT_FOLDER, dir_list[-1])   # load the last saved ckpt

        if ckpt_file is not None:
            previous_model = self.load_checkpoint(ckpt_file, map_location=torch.device('cpu'))
            msg = self.actor_critic.module.load_state_dict(previous_model["state_dict"], strict=False)
            logger.warning(f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')
            logger.info("Loaded previous checkpoint:%s"%ckpt_file)
            start_dagger_it = previous_model['extra_state']['dagger_it']
            start_epoch_it = (int(ckpt_file.split('/')[-1].split('.')[1]) + 1) % self.config.DAGGER.EPOCHS
            if start_epoch_it == 0:
                start_dagger_it += 1

        return start_dagger_it, start_epoch_it

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
        actions=None,
        prog=None,
        rgb_full_global_map=None,
        rgb_frames=None,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                if rgb_frames is not None:
                    rgb_frames.pop(idx)
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[:, state_index]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]
            if actions is not None:
                actions = actions[state_index]
            if prog is not None:
                prog = prog[state_index]
            if rgb_full_global_map is not None:
                rgb_full_global_map = rgb_full_global_map[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

        return (
            envs,
            recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            batch,
            actions,
            prog,
            rgb_full_global_map,
            rgb_frames,
        )

    def eval(self) -> None:
        """Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        """
        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            '', flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                self._eval_checkpoint(self.config.EVAL_CKPT_PATH_DIR, writer)
            else:
                # evaluate multiple checkpoints in order
                num_ckpt = len(os.listdir(self.config.EVAL_CKPT_PATH_DIR))
                prev_ckpt_ind = num_ckpt - 2
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind + 1,
                    )
                    prev_ckpt_ind -= 1

    def _eval_checkpoint(
        self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0, training=False, training_step=0
    ) -> None:
        """Evaluates a single checkpoint. Assumes episode IDs are unique.
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging
        Returns:
            None
        """
        if training:
            checkpoint_path = ''

        finish_process = []
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG and not training:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")["config"]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.NDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.SDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.VLN_ORACLE_WAYPOINT_SENSOR.LAW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.DATASET.split_num = 1
        if config.MODEL.PREDICTION_MONITOR.use:
            config.TASK_CONFIG.TASK.SENSORS.remove('GT_SEMANTIC_MAP_SENSOR')
        config.NUM_PROCESSES = 11 if config.NUM_PROCESSES > 11 else config.NUM_PROCESSES
        config.SIMULATOR_GPU_IDS = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))))

        if training:
            self.actor_critic.module.net.rgb_mapping_module.full_global_map = torch.zeros([config.NUM_PROCESSES] + list(self.actor_critic.module.net.rgb_mapping_module.full_global_map.shape[1:]), device=self.device)
            self.actor_critic.module.net.rgb_mapping_module.agent_view = torch.zeros([config.NUM_PROCESSES] + list(self.actor_critic.module.net.rgb_mapping_module.agent_view.shape[1:]), device=self.device)

        if training:
            config.STOP_CONDITION.TYPE = 'prog'
            config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/val_unseen_min.json.gz'
        # config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/val_unseen.json.gz'
        if len(config.VIDEO_OPTION) > 0 and not training:
            config.SENSORS.append('SEMANTIC_SENSOR')
            config.TASK_CONFIG.TASK.SENSORS.append('SEMANTIC_FILTER_SENSOR')
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        # setup agent
        if self.envs is not None:
            self.envs.close()
            self.envs = None
        self.envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        if not training:
            self._setup_actor_critic(config, not config.random_agent, checkpoint_path)

        observations = self.envs.reset()
        epidsode_reset_flag = True
        observations = transform_obs(
            observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, self.device
        )
        batch = batch_obs(observations, self.device)

        eval_recurrent_hidden_states = torch.zeros(
            self.actor_critic.module.net.num_recurrent_layers,
            config.NUM_PROCESSES,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            config.NUM_PROCESSES, 2, device=self.device
        )
        not_done_masks = torch.zeros(config.NUM_PROCESSES, 1, device=self.device)

        stats_episodes = {}  # dict of dicts that stores stats per episode

        count_step = 0

        rgb_frames = None
        if len(config.VIDEO_OPTION) > 0 and not training:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
            rgb_frames = [[] for _ in range(config.NUM_PROCESSES)]

        pbar = tqdm.tqdm(total=sum(self.envs.number_of_episodes), dynamic_ncols=True, desc="Eval_ckpt_{}".format(str(training_step)))
        self.actor_critic.eval()
        step = 0
        while (
            self.envs.num_envs > 0 and len(stats_episodes) < config.EVAL.EPISODE_COUNT
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                if count_step % config.step_num == 0 and count_step >= 24:
                    (_, actions, _, eval_recurrent_hidden_states) = self.actor_critic.module.act(
                        batch,
                        eval_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=True,
                    )
                else:
                    self.actor_critic.module.update_map(batch, not_done_masks)
                if count_step < 24:
                    actions = batch['waypoint'][:, :2]
                prev_actions.copy_(actions)

            step_inputs = [
                {
                    'action': actions[e].cpu(),
                    'prog': self.actor_critic.module.prog[e].cpu().item() if count_step >= 24 else -1,
                    'epidsode_reset_flag': epidsode_reset_flag ,
                    'depth_img': observations[e]['depth'],
                }
                for e in range(self.envs.num_envs)
            ]
            outputs = self.envs.step(step_inputs) 
            epidsode_reset_flag = False
            step += 1
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            if len(config.VIDEO_OPTION) > 0 and not training:
                for i in range(self.envs.num_envs):
                    observations[i]['ego_map_vis'] = batch['ego_map'][i].cpu().numpy()

            count_step += 1

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            # reset envs and observations if necessary
            for i in range(self.envs.num_envs):
                if len(config.VIDEO_OPTION) > 0 and len(os.listdir(config.VIDEO_DIR)) < config.VIDEO_NUM and not training:
                    att_map = self.actor_critic.module.net.att_map_t_m[i] if count_step-1 >= 24 else torch.zeros(24*24)
                    frame = observations_to_image(observations[i], infos[i], step_inputs[i], att_map)
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                pbar.update()
                stats_episodes[current_episodes[i].episode_id] = infos[i]
                prev_actions[i] = torch.zeros(2)

                finish_process.append(i)
                if len(config.VIDEO_OPTION) > 0 and len(os.listdir(config.VIDEO_DIR)) < config.VIDEO_NUM and not training \
                    and finish_process.count(i) // 3 <= math.ceil(config.VIDEO_NUM / config.NUM_PROCESSES) and finish_process.count(i) % 3 == 1:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            "spl": stats_episodes[current_episodes[i].episode_id]["spl"]
                        },
                        tb_writer=writer,
                    )

                if len(config.VIDEO_OPTION) > 0:
                    del stats_episodes[current_episodes[i].episode_id]["top_down_map"]
                    del stats_episodes[current_episodes[i].episode_id]["collisions"]
                    rgb_frames[i] = []

                if not training:
                    aggregated_stats = {}
                    num_episodes = len(stats_episodes)
                    for stat_key in next(iter(stats_episodes.values())).keys():
                        aggregated_stats[stat_key] = (
                            sum([v[stat_key] for v in stats_episodes.values()]) / num_episodes
                        )
                    logger.info(aggregated_stats)

            if np.array(dones).all():
                self.envs.resume_all()
                observations = self.envs.reset()
                epidsode_reset_flag = True
                count_step = 0
                eval_recurrent_hidden_states = torch.zeros(
                    self.actor_critic.module.net.num_recurrent_layers,
                    config.NUM_PROCESSES,
                    self.config.MODEL.STATE_ENCODER.hidden_size,
                    device=self.device,
                )
                prev_actions = torch.zeros(
                    config.NUM_PROCESSES, 2, device=self.device
                )
                not_done_masks = torch.zeros(config.NUM_PROCESSES, 1, device=self.device)
                self.actor_critic.module.prog = torch.zeros(config.NUM_PROCESSES, 1, device=self.device)
                if self.actor_critic.module.net.rgb_mapping_module is not None:
                    self.actor_critic.module.net.rgb_mapping_module.full_global_map = torch.zeros(
                        config.NUM_PROCESSES,
                        config.MODEL.RGBMAPPING.global_map_size,
                        config.MODEL.RGBMAPPING.global_map_size,
                        config.MODEL.RGBMAPPING.map_depth,
                        device=self.device
                    )
                if len(config.VIDEO_OPTION) > 0:
                    rgb_frames = [[] for _ in range(config.NUM_PROCESSES)]

            observations = transform_obs(
                observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, self.device
            )
            batch = batch_obs(observations, self.device)

            if np.array(dones).all():
                actions = batch['waypoint'][:, :2]

            envs_to_pause = []
            next_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                self.envs,
                eval_recurrent_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
                actions,
                self.actor_critic.module.prog,
                rgb_full_global_map,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                eval_recurrent_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
                actions,
                self.actor_critic.module.prog if count_step >= 24 else None,
                self.actor_critic.module.net.rgb_mapping_module.full_global_map,
                rgb_frames,
            )
            self.actor_critic.module.net.rgb_mapping_module.full_global_map = rgb_full_global_map

        self.envs.close()
        self.envs = None

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()]) / num_episodes
            )

        if not training:
            split = config.TASK_CONFIG.DATASET.SPLIT
            os.makedirs(config.METRIC_DIR, exist_ok=True)
            with open(os.path.join(config.METRIC_DIR, f"stats_ckpt_{checkpoint_index}_{split}.json"), "w") as f:
                json.dump(aggregated_stats, f, indent=4)
            with open(os.path.join(config.METRIC_DIR, f"each_stat_ckpt_{checkpoint_index}_{split}.json"), "w") as f:
                json.dump(stats_episodes, f)

        if not training:
            logger.info(f"Episodes evaluated: {num_episodes}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)
        else:
            for k, v in aggregated_stats.items():
                logger.info(f"Eval while training average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_while_training_{k}", v, training_step)
            writer.flush()

    def empty_cuda_cache(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        gc.collect()

    def change_data_type(self, traj_obs):
        for k, v in traj_obs.items():
            traj_obs[k] = v.numpy()
            if k == 'vln_oracle_action_sensor':
                traj_obs[k] = traj_obs[k].astype(np.uint8)
            if k == 'rgb_ego_map':
                traj_obs[k] = traj_obs[k].astype(np.float16)
            if k == 'gt_path':
                traj_obs[k] = traj_obs[k].astype(np.float16)
            if k == 'rgb':
                traj_obs[k] = traj_obs[k].astype(np.uint8)
            if k == 'depth':
                traj_obs[k] = traj_obs[k].astype(np.float16)
            if k == 'rgb_features':
                traj_obs[k] = traj_obs[k].astype(np.float16)
            if k == 'depth_features':
                traj_obs[k] = traj_obs[k].astype(np.float16)
            if k == 'gt_semantic_map':
                traj_obs[k] = traj_obs[k].astype(np.int)

    def inference(self) -> None:
        pass
