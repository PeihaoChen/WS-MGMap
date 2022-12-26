import numpy as np
from typing import Optional

import habitat
from habitat import Config, Dataset
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat

from vlnce_baselines.common.action_maker import GTMapActionMaker, DDPPOActionMaker


@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.config = config
        self.device = self._env._config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE

        self.follower = ShortestPathFollowerCompat(self._env.sim, 0.5, return_one_hot=False)
        self.follower.mode = 'geodesic_path'
        self.steppppp = 0

        if self.config.use_ddppo:
            self.ddppo_action_maker = DDPPOActionMaker(config, self._env)
        else:
            self.gt_map_action_maker = GTMapActionMaker(config)

    def reset(self):
        observation = super(VLNCEDaggerEnv, self).reset()
        return observation

    def step(self, action, prog, epidsode_reset_flag=None, depth_img=None):
        if self.config.use_ddppo and epidsode_reset_flag is True:
            self.ddppo_action_maker.l_policy.reset()
            self.ddppo_action_maker.sg_reset()
            self.steppppp = 0

        agent_state = self._env._sim.get_agent_state()
        if self.config.use_ddppo:
            self.waypoint = self.ddppo_action_maker.preprocess(action, agent_state)
            action_choice = self.ddppo_action_maker.action_decision(self.steppppp, self.waypoint, depth_img)
        else:
            self.waypoint = self.gt_map_action_maker.preprocess(action, agent_state)
            action_choice = self.gt_map_action_maker.action_decision(self.waypoint, self.follower)

        stop = self.decide_stop(prog)
        if stop:
            action_choice = 0

        if self._env._elapsed_steps < 24:
            action_choice = 2

        observation, reward, done, info = self.step_bak(action_choice)

        self.steppppp += 1

        return observation, reward, done, info

    def step_bak(self, action):
        observations, reward, done, info = super().step(action)
        return observations, reward, done, info

    def decide_stop(self, prog):
        if prog == -1 and self._distance_waypoint(self._env.current_episode.goals[0].position) < 0.5:
            return True
        elif prog > self.config.STOP_CONDITION.PROG_THRESHOLD:
            return True
        return False

    def _distance_waypoint(self, waypoint):
        agent_position = self._env._sim.get_agent_state().position
        return self._env.sim.geodesic_distance(waypoint, agent_position)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        return self._env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(VLNCEDaggerEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        return self._env.episode_over

    def get_info(self, observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
