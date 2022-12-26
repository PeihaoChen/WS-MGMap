import os
import cv2
import gzip
import json
import numpy as np
from gym import spaces
from typing import Any

import torch
import torch.nn.functional as F

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import maps
from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat

from vlnce_baselines.common.rgb_mapping import get_grid
from vlnce_baselines.common.action_maker import TransfomationRealworldAgent


@registry.register_sensor
class VLNOracleActionSensor(Sensor):
    """Sensor for observing the optimal action to take. The assumption this
    sensor currently makes is that the shortest path to the goal is the
    optimal path.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        super().__init__(config=config)

        # all goals can be navigated to within 0.5m.
        goal_radius = getattr(config, "GOAL_RADIUS", 0.5)
        if config.USE_ORIGINAL_FOLLOWER:
            self.follower = ShortestPathFollowerCompat(
                sim, goal_radius, return_one_hot=False
            )
            self.follower.mode = "geodesic_path"
        else:
            self.follower = ShortestPathFollower(sim, goal_radius, return_one_hot=False)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "vln_oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [best_action if best_action is not None else HabitatSimActions.STOP]
        )


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    """Sensor for observing how much progress has been made towards the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "progress"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        # TODO: what is the correct sensor type?
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        distance_from_start = episode.info["geodesic_distance"]
        progress = (distance_from_start - distance_to_target) / distance_from_start
        return np.array([progress])


@registry.register_sensor
class VLNOracleWaypointSensor(Sensor):
    """Sensor for waypoint towards the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        super().__init__(config=config)
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._map_size = config.MAP_SIZE

        goal_radius = getattr(config, "GOAL_RADIUS", 0.5)
        if config.USE_ORIGINAL_FOLLOWER:
            self.follower = ShortestPathFollowerCompat(
                sim, goal_radius, return_one_hot=False
            )
            self.follower.mode = "geodesic_path"
        else:
            self.follower = ShortestPathFollower(sim, goal_radius, return_one_hot=False)
        self._sim = sim

        self.use_law = config.LAW.USE
        gt_path = config.LAW.GT_PATH.format(split=config.LAW.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_waypoint_locations = json.load(f)
        self.is_sparse = config.LAW.IS_SPARSE
        self.num_inter_waypoints = config.LAW.NUM_WAYPOINTS

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "waypoint"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(4,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        if self.use_law:
            goal_pos = self.get_goal(episode)
        else:
            goal_pos = episode.goals[0].position
        points = self._sim.get_straight_shortest_path_points(agent_position, goal_pos)
        if len(points) < 2:
            return None

        waypoint = self.get_waypoint(points)

        self.trans_tool = TransfomationRealworldAgent(self._sim.get_agent_state())
        wp_a = self.trans_tool.realworld2agent(waypoint)

        resolution = (self._coordinate_max - self._coordinate_min) / self._map_resolution[0]
        wp_ego_x = (wp_a[0] / resolution).astype(np.int)  
        wp_ego_y = (-wp_a[2] / resolution).astype(np.int)
        wp_norm_x = wp_ego_x / (self._map_size // 2)
        wp_norm_y = wp_ego_y / (self._map_size // 2)

        return np.array([wp_norm_x, wp_norm_y])

    def get_goal(self, episode):
        if self.num_inter_waypoints > 0:
            locs = self.gt_waypoint_locations[str(episode.episode_id)]["locations"]
            ep_path_length = self._sim.geodesic_distance(locs[0], episode.goals[0].position)

            way_locations = [locs[0]]
            count = 0
            dist = ep_path_length / (self.num_inter_waypoints+1)
            for way in locs[:-1]:
                d = self._sim.geodesic_distance(locs[0], way)
                if d >= dist:
                    way_locations.append(way)
                    if count >= (self.num_inter_waypoints-1):
                        break
                    count += 1
                    dist += ep_path_length / (self.num_inter_waypoints+1)

            way_locations.append(episode.goals[0].position)
        else:
            if self.is_sparse:
                # Sparse supervision of waypoints
                way_locations = episode.reference_path
            else:
                # Dense supervision of waypoints
                way_locations = self.gt_waypoint_locations[str(episode.episode_id)]["locations"]

        current_position = self._sim.get_agent_state().position.tolist()
        nearest_dist = float("inf")
        nearest_way = way_locations[-1]

        for ind, way in reversed(list(enumerate(way_locations))):
            distance_to_way = self._sim.geodesic_distance(current_position, way)

            if distance_to_way >= 3.0 and distance_to_way < nearest_dist:
                dist_way_to_goal = self._sim.geodesic_distance(way, episode.goals[0].position)
                dist_agent_to_goal = self._sim.geodesic_distance(current_position, episode.goals[0].position)

                if dist_agent_to_goal > dist_way_to_goal:
                    nearest_dist = distance_to_way
                    nearest_way = way

        return nearest_way

    def get_waypoint(self, points):
        path_line = np.zeros(self._map_resolution, dtype=np.uint8)
        for index in range(len(points) - 1):
            x_t_1, y_t_1 = maps.to_grid(
                points[index][0], points[index][2],
                self._coordinate_min, self._coordinate_max, self._map_resolution,
            )
            x_t_2, y_t_2 = maps.to_grid(
                points[index + 1][0], points[index + 1][2],
                self._coordinate_min, self._coordinate_max, self._map_resolution,
            )
            cv2.line(path_line, (y_t_1, x_t_1), (y_t_2, x_t_2), 255, 1)

        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        fog_line = np.zeros(self._map_resolution, dtype=np.uint8)
        cv2.circle(fog_line, (a_y, a_x), 20, 255, 2)

        searched = []
        def search(point):
            searched.append([point[0], point[1]])
            if fog_line[point[0], point[1]]:
                return point
            for p in [(-1,0), (0,-1), (1,0), (0,1), (-1,-1), (1,-1), (1,1), (-1,1)]:
                if path_line[point[0]+p[0], point[1]+p[1]] and [point[0]+p[0], point[1]+p[1]] not in searched:
                    s_point = search([point[0]+p[0], point[1]+p[1]])
                    if s_point is None:
                        continue
                    return s_point

        cross_line = np.where((path_line & fog_line) != 0)
        if cross_line[0].shape[0] > 0:
            frontier = search([a_x, a_y])
            if frontier is None:
                frontier = [cross_line[0][0], cross_line[1][0]]
            frontier = maps.from_grid(
                frontier[0], frontier[1],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )
        else:
            frontier = [points[-1][0], points[-1][2]]

        waypoint = np.array([frontier[0], points[0][1], frontier[1]])
        return waypoint


@registry.register_sensor
class VLNOraclePathSensor(Sensor):
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._map_size = config.MAP_SIZE

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gt_path"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(100, 100), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        goal_pos = episode.goals[0].position
        points = self._sim.get_straight_shortest_path_points(agent_position, goal_pos)
        if len(points) < 2:
            return None
        gt_path = self.get_gt_path(points)
        return gt_path

    def get_gt_path(self, points):
        path_line = np.zeros([self._map_size, self._map_size])
        self.trans_tool = TransfomationRealworldAgent(self._sim.get_agent_state())

        for index in range(len(points) - 1):
            resolution = (self._coordinate_max - self._coordinate_min) / self._map_resolution[0]

            a1 = self.trans_tool.realworld2agent(points[index])
            x_t_1 = (a1[2] / resolution + self._map_size // 2).astype(np.int)
            y_t_1 = (a1[0] / resolution + self._map_size // 2).astype(np.int)

            a2 = self.trans_tool.realworld2agent(points[index + 1])
            x_t_2 = (a2[2] / resolution + self._map_size // 2).astype(np.int)
            y_t_2 = (a2[0] / resolution + self._map_size // 2).astype(np.int)

            cv2.line(path_line, (y_t_1, x_t_1), (y_t_2, x_t_2), 255, self.config.LINE_WIDTH)

        waypoint_dis = path_line / 255
        line_point_x, line_point_y = np.where(waypoint_dis != 0)
        line_point = np.concatenate([line_point_x[np.newaxis, :], line_point_y[np.newaxis, :]], axis=0)
        line_point = np.repeat(line_point[np.newaxis, :, :], 100, axis=0)
        line_point = np.repeat(line_point[np.newaxis, :, :, :], 100, axis=0)

        x, y = np.linspace(0, 99, 100), np.linspace(0, 99, 100)
        xv, yv = np.meshgrid(x, y)
        all_point = np.concatenate([yv[:, :, np.newaxis], xv[:, :, np.newaxis]], axis=2)
        all_point = np.repeat(all_point[:, :, :, np.newaxis], line_point.shape[-1], axis=3)

        dis_map = np.min(np.sqrt(np.sum((all_point - line_point)**2, axis=2)), axis=2)

        return dis_map


@registry.register_sensor
class SemanticFilterSensor(Sensor):
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        super().__init__(config=config)
        self.sim = sim
        self.prev_episode_id = None
        self.label_to_27 = np.array([
            0, 15, 17, 1, 2, 3, 18, 19, 4, 15, 5, 6, 16, 20, 7, 8, 17,
            17, 9, 21, 22, 16, 10, 11, 15, 12, 13, 23, 16, 16, 16, 16,
            16, 24, 25, 16, 16, 14, 26, 16, 16,
        ])

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "semantic_filter"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.HEIGHT, self.config.WIDTH, self.config.CATEGORY),
            dtype=np.float,
        )

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        semantic = observations['semantic']

        if self.prev_episode_id != episode.episode_id:
            scene = self.sim.semantic_annotations()
            instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
            self.mapping = np.array([instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id))])
        self.prev_episode_id = episode.episode_id

        semantic = np.take(self.mapping, semantic)
        semantic[semantic == -1] = 0
        semantic = np.take(self.label_to_27, semantic)
        h, w = semantic.shape
        semantic_filter = np.eye(27, dtype=np.float32)[semantic.reshape(-1)].reshape(h, w, 27)

        return semantic_filter


@registry.register_sensor
class GtSemanticMapSensor(Sensor):
    r"""Sensor for generating semantic map grounth truth
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self.gt_path = 'data/map_data/semantic/{}'.format(config.SPLIT)
        self.half_size = config.MAP_SIZE // 2
        self.prev_episode_id = None
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "gt_semantic_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=27.0, shape=(100, 100), dtype=np.long)

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        if self.prev_episode_id != episode.episode_id:
            self.init_agent_state = self._sim.get_agent_state()

            self.global_gt_semmap = np.load(os.path.join(self.gt_path, 'ep_'+str(episode.episode_id)+'.npy'))
            self.global_gt_semmap = torch.from_numpy(self.global_gt_semmap).unsqueeze(0).unsqueeze(0).float()

            rever_pose = torch.FloatTensor([0, 0, self._sim.record_heading]).unsqueeze(0)
            rot_mat, _ = get_grid(rever_pose, self.global_gt_semmap.size(), 'cpu')
            self.global_gt_semmap = F.grid_sample(self.global_gt_semmap, rot_mat, mode='nearest')

        agent_state = self._sim.get_agent_state()
        grid_y = (agent_state.position[0] - self.init_agent_state.position[0]) / 0.12 + 240
        grid_x = (agent_state.position[2] - self.init_agent_state.position[2]) / 0.12 + 240
        st_pose = torch.FloatTensor([
            (grid_y - (480//2)) / (480//2),
            (grid_x - (480//2)) / (480//2),
            - self._sim.record_heading,
        ]).unsqueeze(0)

        rot_mat, tra_mat = get_grid(st_pose, self.global_gt_semmap.size(), 'cpu')
        transed_map = F.grid_sample(self.global_gt_semmap, tra_mat, mode='nearest')
        rotated_map = F.grid_sample(transed_map, rot_mat, mode='nearest')
        rotated_map = F.pad(rotated_map, (self.half_size, self.half_size, self.half_size, self.half_size), 'constant', 0)

        self.prev_episode_id = episode.episode_id

        return rotated_map.squeeze()[289-self.half_size: 289+self.half_size, 289-self.half_size: 289+self.half_size].long()

@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(quat, direction_vector)
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        heading = self._quat_to_xy_heading(rotation_world_agent.inverse())
        self._sim.record_heading = heading

        return heading
