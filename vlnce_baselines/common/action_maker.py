import torch

from habitat.utils.visualizations import maps

from vlnce_baselines.common.utils import TransfomationRealworldAgent
from vlnce_baselines.models.ddppo_policy import DdppoPolicy, SemanticGrid, utils


class ActionMaker():
    def __init__(self, config) -> None:
        self.config = config
        self.ego_map_size = config.ego_map_size
        self.map_range_max = maps.COORDINATE_MAX
        self.map_range_min = maps.COORDINATE_MIN
        self.map_size = 1250

    def preprocess(self, action, agent_state):
        resolution = (self.map_range_max - self.map_range_min) / self.map_size
        tran_real_to_agent = TransfomationRealworldAgent(agent_state)

        waypoint_norm = torch.tanh(action)
        waypoint_a = torch.zeros([3])
        waypoint_a[0] = waypoint_norm[0] * (self.ego_map_size / 2) * resolution
        waypoint_a[2] = -waypoint_norm[1] * (self.ego_map_size / 2) * resolution

        waypoint_w = tran_real_to_agent.agent2realworld(waypoint_a)

        return waypoint_w

    def action_decision(self) -> int:
        pass


class GTMapActionMaker(ActionMaker):
    def __init__(self, config) -> None:
        super().__init__(config)

    def action_decision(self, goal, follower) -> int:
        action = follower.get_next_action(goal)

        if action is None:
            action = 1

        return action


class DDPPOActionMaker(ActionMaker):
    def __init__(self, config, _env) -> None:
        super().__init__(config)
        self.utils = utils()
        self._env = _env
        self.device = torch.device("cuda", self._env._config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID)
        self.grid_dim = (192, 192)
        self.global_dim = (512,512)
        self.heatmap_size = 24
        self.cell_size = 0.05
        self.img_size = (256, 256)
        self.n_object_classes = 27
        self.n_spatial_classes = 3
        model_path = 'data/pretrain_model/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth'
        self.l_policy = DdppoPolicy(path=model_path)
        self.l_policy = self.l_policy.to(self.device)
        self.sg_reset()

    def sg_reset(self):
        self.sg_global = SemanticGrid(
            1, self.global_dim, self.heatmap_size, self.cell_size,
            spatial_labels=self.n_spatial_classes, object_labels=self.n_object_classes,
            device=self.device,
        )
        self.abs_poses = []
        self.agent_height = []
        # long term goal in global grid map
        self.ltg_abs_coords = torch.zeros((1, 1, 2), dtype=torch.int64).to(self.device)
        self.ltg_abs_coords_list = []

    def run_local_policy(self, depth, goal, pose_coords, rel_agent_o, step):
        planning_goal = goal.squeeze(0).squeeze(0)
        planning_pose = pose_coords.squeeze(0).squeeze(0)

        sq = torch.square(planning_goal[0]-planning_pose[0]) + torch.square(planning_goal[1]-planning_pose[1])
        rho = torch.sqrt(sq.float())
        phi = torch.atan2(((planning_pose[0]-planning_goal[0]).float()), (planning_pose[1]-planning_goal[1]).float())
        phi = phi - rel_agent_o
        rho = rho * self.cell_size

        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32).to(self.device)
        depth = depth.reshape(self.img_size[0], self.img_size[1], 1)
        return self.l_policy.plan(depth, point_goal_with_gps_compass, step)

    def transform_waypoint2cm2(self, t, ltg):
        ltg_cm2 = [] 
        ltg_cm2.append(-ltg[2])
        ltg_cm2.append(-ltg[0])

        agent_state = self._env.sim.get_agent_state()
        agent_pose, y_height = self.utils.get_sim_location(agent_state) 
        ltg_cm2.append(agent_pose[2])
        self.abs_poses.append(agent_pose)
        self.agent_height.append(y_height)

        rel_abs_pose = self.utils.get_rel_pose(self.abs_poses[t], self.abs_poses[0])  
        _rel_abs_pose = torch.Tensor(rel_abs_pose).unsqueeze(0).float()
        _rel_abs_pose = _rel_abs_pose.to(self.device)
        abs_pose_coords = self.utils.get_coord_pose(self.sg_global, _rel_abs_pose, self.abs_poses[0], self.global_dim[0], self.cell_size, self.device) # B x T x 3

        rel_ltg_abs_pose = self.utils.get_rel_pose(pos2=ltg_cm2, pos1=self.abs_poses[0])  
        _rel_ltg_abs_pose = torch.Tensor(rel_ltg_abs_pose).unsqueeze(0).float()
        _rel_ltg_abs_pose = _rel_ltg_abs_pose.to(self.device)
        ltg_coords = self.utils.get_coord_pose(self.sg_global, _rel_ltg_abs_pose, self.abs_poses[0], self.global_dim[0], self.cell_size, self.device)

        return ltg_coords, abs_pose_coords, rel_abs_pose

    def action_decision(self, t, ltg, depth):
        ltg_abs_coords, abs_pose_coords, rel_abs_pose = self.transform_waypoint2cm2(t, ltg)
        depth = torch.tensor(depth).to(self.device)
        action_id = self.run_local_policy(
            depth=depth,
            goal=ltg_abs_coords.clone(),
            pose_coords=abs_pose_coords.clone(), 
            rel_agent_o=rel_abs_pose[2],
            step=t,
        )
        return action_id
