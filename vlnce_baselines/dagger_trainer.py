import os
import time
import random
import time
import psutil
import zlib
import tqdm
import msgpack_numpy
from collections import defaultdict
from multiprocessing import Pool

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs

from vlnce_baselines.common_trainer import CommonTrainer
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.utils import transform_obs


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()
        return self


def compress_data(data):
    return zlib.compress(msgpack_numpy.packb(data, use_bin_type=True))


def collate_fn(batch):
    """Each sample in batch: (
            obs,
            prev_actions,
            oracle_actions,
            inflec_weight,
        )
    return: 
        observation_batch: {
            'instruction': [TxN, 200]
            'rgb_features': [TxN, 2048, 4, 4]
            ...
        }
        prev_actions_batch: [TxN, 1]
        corrected_actions_batch: [T, N]
        ...
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount <= 0:
            return t[:max_len]

        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations_batch[bid][sensor])

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    limited_len_by_gpu = 200
    max_traj_len = min(max_traj_len, limited_len_by_gpu)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        corrected_actions_batch[bid] = _pad_helper(corrected_actions_batch[bid], max_traj_len)
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(weights_batch, dtype=torch.float)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 2),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)
    return [ele for block in blocks for ele in block]


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
        rank=-1,
        world_size=-1
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 1
        self._preload = []
        self._preload_index = []
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.num_loaded_data = 0

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    load_index = self.load_ordering.pop()
                    preload_data = msgpack_numpy.unpackb(
                        zlib.decompress(txn.get(str(load_index).encode())), raw=False
                    )
                    if 'ep_id' in preload_data[0].keys():
                        del preload_data[0]['ep_id']
                    new_preload.append(preload_data)

                    lengths.append(len(new_preload[-1][-1]))
                    self._preload_index.append(load_index)

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in sorted_ordering:
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))
        inflections = torch.zeros_like(obs['progress'].squeeze(-1)).numpy().tolist()

        return (obs, prev_actions, oracle_actions, self.inflec_weights[inflections])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            per_proc = int(np.floor(self.length / self.world_size))

            start = per_proc * self.rank
            end = min(start + per_proc, self.length)
            assert end - start == per_proc 
        else:
            per_proc = int(np.floor(self.length / self.world_size))
            per_worker = int(np.floor(per_proc / worker_info.num_workers))

            start = per_worker * worker_info.id + per_proc * self.rank
            end = min(start + per_worker, self.length)
            assert end - start == per_worker 

        self.load_ordering = list(
            reversed(_block_shuffle(list(range(start, end)), self.preload_size))
        )
        if self.rank == 0:
            logger.info(f'[Dataset]: Total data size {self.length}')

        return self
    
    def __len__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return int(np.floor(self.length / self.world_size))
        else:
            per_proc = int(np.floor(self.length / self.world_size))
            return int(np.floor(per_proc / worker_info.num_workers)) * worker_info.num_workers


@baseline_registry.register_trainer(name="dagger")
class DaggerTrainer(CommonTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.lmdb_features_dir = self.config.DAGGER.LMDB_FEATURES_DIR.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )

    def _update_dataset(self, data_it):
        self.empty_cuda_cache()

        if self.envs is None:
            allocated_cuda_memory = torch.cuda.memory_allocated(device=self.local_rank) / (1024 * 1024 * 1024)
            if allocated_cuda_memory > 6:
                self.config.defrost()
                self.config.NUM_PROCESSES = int((12 - allocated_cuda_memory) // 2.5)
                self.config.freeze()
                logger.info("cuda memory is not enough, processes reduce to ", int((12 - allocated_cuda_memory) // 2.5))
            self.config.defrost()
            self.config.TASK_CONFIG.DATASET.split_num = self.world_size
            self.config.TASK_CONFIG.DATASET.split_rank = self.local_rank
            self.config.freeze()
            self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        else:
            self.envs.resume_all()

        recurrent_hidden_states = torch.zeros(
            self.actor_critic.module.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 2, device=self.device
        )
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)

        observations = self.envs.reset()
        observations = transform_obs(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, self.device
        )
        batch = batch_obs(observations, self.device)

        episodes = [[] for _ in range(self.envs.num_envs)]
        skips = [False for _ in range(self.envs.num_envs)]
        dones = [False for _ in range(self.envs.num_envs)]

        count_step = 0
        each_count_step = [0 for _ in range(self.envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        if self.config.DAGGER.P == 0.0:
            # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
            beta = 0.0
        else:
            beta = self.config.DAGGER.P ** data_it

        ensure_unique_episodes = beta == 1.0

        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())
            return hook

        rgb_features = None
        rgb_hook = None
        rgb_features = torch.zeros((1,), device="cpu")
        rgb_hook = self.actor_critic.module.net.rgb_encoder.base_model.layer4_1x1.register_forward_hook(
            hook_builder(rgb_features)
        )

        depth_features = None
        depth_hook = None
        depth_features = torch.zeros((1,), device="cpu")
        depth_hook = self.actor_critic.module.net.depth_encoder.visual_encoder.register_forward_hook(
            hook_builder(depth_features)
        )

        rgb_ego_map = None
        rgb_ego_map_hook = None
        rgb_ego_map = torch.zeros((1,), device="cpu")
        rgb_ego_map_hook = self.actor_critic.module.net.rgb_mapping_module.register_forward_hook(
            hook_builder(rgb_ego_map)
        )

        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = set(
                [ep.episode_id for ep in self.envs.current_episodes()]
            )

        def writeCache(cache):
            with Pool(8) as workers:
                cache = workers.map(compress_data, cache)
            txn = lmdb_env.begin(write=True)
            existed_size = lmdb_env.stat()["entries"]
            for i, v in enumerate(cache):
                txn.put(str(existed_size + i).encode(), v)
            txn.commit()

        torch.distributed.barrier()
        time.sleep(1*self.local_rank)
        cache = []
        with lmdb.open(self.lmdb_features_dir, map_size=int(self.config.DAGGER.LMDB_MAP_SIZE)) as lmdb_env, torch.no_grad():

            required_size = (data_it+1) * self.config.DAGGER.UPDATE_SIZE
            remain_update_size = required_size - lmdb_env.stat()["entries"]
            start_id = lmdb_env.stat()["entries"]       
            if self.local_rank == 0:
                pbar = tqdm.tqdm(total=remain_update_size, smoothing=0.01, desc=f"Collecting Data, DAgger iter {data_it}")
            
            while collected_eps < remain_update_size and not (lmdb_env.stat()["entries"] > required_size):
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = self.envs.current_episodes()

                for i in range(self.envs.num_envs):
                    if dones[i] and not skips[i]:
                        if len(episodes[i]) > self.config.ep_max_len or len(episodes[i]) < 25:
                            episodes[i] = []
                            continue
                        ep = episodes[i]
                        ep = [ep[sp_num] for sp_num in range(24, len(ep), self.config.step_num)]
                        traj_obs = batch_obs(
                            [step[0] for step in ep], device=torch.device("cpu")
                        )
                        if 'heading' in traj_obs.keys():
                            del traj_obs['heading']
                        if 'compass' in traj_obs.keys():
                            del traj_obs['compass']
                        if 'gps' in traj_obs.keys():
                            del traj_obs['gps']
                        self.change_data_type(traj_obs)
                        if ensure_unique_episodes:
                            traj_obs['ep_id'] = current_episodes[i].episode_id

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep]),
                            np.array([step[2] for step in ep]),
                        ]

                        cache.append(transposed_ep)
                        if self.local_rank == 0:
                            pbar.update(lmdb_env.stat()["entries"] - start_id - pbar.n)
                        collected_eps += 1

                        ava_mem = float(psutil.virtual_memory().available) / 1024 / 1024 / 1024
                        if (len(cache) % self.config.DAGGER.LMDB_COMMIT_FREQUENCY == 0 or ava_mem < 10) and len(cache) != 0:
                            writeCache(cache)
                            del cache
                            cache = []

                        if ensure_unique_episodes:
                            if current_episodes[i].episode_id in ep_ids_collected:
                                envs_to_pause.append(i)
                            else:
                                ep_ids_collected.add(current_episodes[i].episode_id)
                                with open(os.path.join(self.lmdb_features_dir, 'collected_ep.txt'), 'a') as fp:
                                    fp.write(f'{current_episodes[i].episode_id}\n')

                    if dones[i]:
                        episodes[i] = []
                        each_count_step[i] = 0

                if ensure_unique_episodes:
                    (
                        self.envs,
                        recurrent_hidden_states,
                        not_done_masks,
                        prev_actions,
                        batch, _, _, _, _
                    ) = self._pause_envs(
                        envs_to_pause,
                        self.envs,
                        recurrent_hidden_states,
                        not_done_masks,
                        prev_actions,
                        batch,
                    )
                    if self.envs.num_envs == 0:
                        break

                if count_step % self.config.step_num == 0:
                    (_, actions, _, recurrent_hidden_states) = self.actor_critic.module.act(
                        batch,
                        recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=True,
                    )
                else:
                    self.actor_critic.module.update_map(batch, not_done_masks)

                count_step += 1
                each_count_step = [i + 1 for i in each_count_step]
                for i in range(self.envs.num_envs):
                    if each_count_step[i] == 23:
                        recurrent_hidden_states[:, i, :] *= 0

                for i in range(self.envs.num_envs):
                    if torch.rand(1, dtype=torch.float, device=self.device) < beta:
                        actions[i] = batch['waypoint'][i]

                for i in range(self.envs.num_envs):
                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"]

                    if depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]

                    if rgb_ego_map is not None:
                        observations[i]["rgb_ego_map"] = rgb_ego_map[i]

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].cpu().numpy(),
                            batch["waypoint"][i].cpu().numpy(),
                        )
                    )

                step_inputs = [
                    {
                        'action': actions[e].cpu(),
                        'prog': -1,
                    }
                    for e in range(self.envs.num_envs)
                ]
                outputs = self.envs.step(step_inputs)
                observations, rewards, dones, _ = [list(x) for x in zip(*outputs)]

                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

                observations = transform_obs(
                    observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, self.device
                )
                batch = batch_obs(observations, self.device)

            writeCache(cache)

        self.envs.close()
        self.envs = None

        if rgb_hook is not None:
            rgb_hook.remove()
        if depth_hook is not None:
            depth_hook.remove()
        if rgb_ego_map_hook is not None:
            rgb_ego_map_hook.remove()
        logger.info("Finish collecting data!")

    def _update_agent(
        self, observations, prev_actions, not_done_masks, corrected_actions, weights
    ):
        T, N = corrected_actions.size()[:2]
        self.optimizer.zero_grad()

        recurrent_hidden_states = torch.zeros(
            self.actor_critic.module.net.num_recurrent_layers,
            N,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        # weights[:24, :] = 0

        AuxLosses.clear()

        pred, aux_loss = self.actor_critic(
            observations, recurrent_hidden_states, prev_actions, not_done_masks, weights
        )

        logits = torch.tanh(pred)
        logits = logits.view(T, N, -1)
        action_loss = F.mse_loss(
            logits, observations['waypoint'][:, :2].view(T, N, -1), reduction="none"
        ).sum(dim=2)
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        loss = action_loss + aux_loss
        loss.backward()

        self.optimizer.step()

        if isinstance(aux_loss, torch.Tensor):
            return loss.item(), action_loss.item(), aux_loss.item()
        else:
            return loss.item(), action_loss.item(), aux_loss

    def train(self) -> None:
        """Main method for training DAgger.
        Returns:
            None
        """               
        if self.local_rank == 0:
            os.makedirs(self.lmdb_features_dir, exist_ok=True)
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        torch.distributed.barrier()

        single_proc_config = self.config.clone()
        single_proc_config.defrost()
        single_proc_config.NUM_PROCESSES = 1
        single_proc_config.freeze()
        self.envs = construct_envs(single_proc_config, get_env_class(self.config.ENV_NAME))

        self._setup_actor_critic(self.config, self.config.DAGGER.LOAD_FROM_CKPT, self.config.DAGGER.CKPT_TO_LOAD)
        start_dagger_it, start_epoch_it = self.resume_dagger()

        self.envs.close()
        del self.envs
        self.envs = None

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs, purge_step=0
        ) as writer:
            for dagger_it in range(start_dagger_it, self.config.DAGGER.ITERATIONS):
                step_id = 0
                if not self.config.DAGGER.PRELOAD_LMDB_FEATURES:
                    self._update_dataset(dagger_it)
                torch.distributed.barrier()
                self.empty_cuda_cache()

                dataset = IWTrajectoryDataset(
                    self.lmdb_features_dir,
                    self.config.DAGGER.USE_IW,
                    inflection_weight_coef=self.config.MODEL.inflection_weight_coef,
                    lmdb_map_size=self.config.DAGGER.LMDB_MAP_SIZE,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                    rank=self.local_rank,
                    world_size=self.world_size
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=4,
                    sampler=None
                )

                AuxLosses.activate()
                for epoch in range(start_epoch_it, self.config.DAGGER.EPOCHS):
                    if self.local_rank == 0:
                        logger.info(f"Start training for Epoch {epoch}")
                        iter_bar = tqdm.tqdm(total=len(diter), leave=False, smoothing=0.05, desc=f"DAgger iter {dagger_it}, Epoch {epoch}")
                        logger.info("start clean")
                        logger.info(f'{self.lmdb_features_dir}/data.mdb')
                        fd = os.open(f'{self.lmdb_features_dir}/data.mdb', os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        logger.info("end clean")                    
                    for iter, batch in enumerate(diter):
                        (
                            observations_batch,
                            prev_actions_batch,
                            not_done_masks,
                            corrected_actions_batch,
                            weights_batch,
                        ) = batch
                        observations_batch = {
                            k: v.float().to(device=self.device, non_blocking=True)
                            for k, v in observations_batch.items()
                        }

                        loss, action_loss, aux_loss = self._update_agent(
                            observations_batch,
                            prev_actions_batch.to(device=self.device, non_blocking=True),
                            not_done_masks.to(device=self.device, non_blocking=True),
                            corrected_actions_batch.to(device=self.device, non_blocking=True),
                            weights_batch.to(device=self.device, non_blocking=True),
                        )

                        if self.local_rank == 0:
                            iter_bar.update()
                            if step_id % self.config.LOG_INTERVAL == 0 and step_id != 0:
                                writer.add_scalar(f"train_loss_iter_{dagger_it}", loss, step_id)
                                writer.add_scalar(f"train_action_loss_iter_{dagger_it}", action_loss, step_id)
                                writer.add_scalar(f"train_aux_loss_iter_{dagger_it}", aux_loss, step_id)
                                writer.flush()
                        step_id += 1

                    if self.local_rank == 0:
                        self.save_checkpoint(
                            f"ckpt.{dagger_it * self.config.DAGGER.EPOCHS + epoch}.pth",
                            extra_state={'dagger_it': dagger_it},
                        )
                    logger.info(f"Local rank {self.local_rank} finish Epoch {epoch}")
                    torch.distributed.barrier()

                    if self.config.DAGGER.EPOCHS > 10:
                        AuxLosses.deactivate()
                        self.empty_cuda_cache()
                        if epoch % 3 == 0 and self.local_rank == 0:
                            self.actor_critic.eval()
                            self._eval_checkpoint('', writer, 0, training=True, training_step=epoch)
                            self.actor_critic.train()
                            self.actor_critic.module.net.depth_encoder.eval()
                            self.actor_critic.module.net.rgb_encoder.eval()
                        torch.distributed.barrier()
                        self.empty_cuda_cache()
                        AuxLosses.activate()

                AuxLosses.deactivate()

                self.empty_cuda_cache()
                if dagger_it % 1 == 0 and self.local_rank == 0:
                    self.actor_critic.eval()
                    self._eval_checkpoint('', writer, 0, training=True, training_step=dagger_it)
                    self.actor_critic.train()
                    self.actor_critic.module.net.depth_encoder.eval()
                    self.actor_critic.module.net.rgb_encoder.eval()
                torch.distributed.barrier()
                self.empty_cuda_cache()
                if self.actor_critic.module.net.rgb_mapping_module is not None:
                    self.actor_critic.module.net.rgb_mapping_module.full_global_map = torch.zeros(
                        [self.config.NUM_PROCESSES] + \
                            list(self.actor_critic.module.net.rgb_mapping_module.full_global_map.shape[1:]),
                        device=self.device
                    )
                    self.actor_critic.module.net.rgb_mapping_module.agent_view = torch.zeros(
                        [self.config.NUM_PROCESSES] + \
                            list(self.actor_critic.module.net.rgb_mapping_module.agent_view.shape[1:]),
                        device=self.device
                    )
