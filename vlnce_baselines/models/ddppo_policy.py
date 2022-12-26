import math
import numpy as np
import quaternion
from gym.spaces import Box, Dict, Discrete
from gym.spaces.box import Box

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy


class DdppoPolicy(nn.Module):
    def __init__(self, path):
        super().__init__()
        spaces = {
            'pointgoal_with_gps_compass': Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
            'depth': Box(
                low=0,
                high=1,
                shape=(256, 256, 1),
                dtype=np.float32,
            )
        }
        observation_space = Dict(spaces)
        action_space = Discrete(4)

        checkpoint = torch.load(path)
        self.hidden_size = checkpoint['model_args'].hidden_size
        # The model must be named self.actor_critic to make the namespaces correct for loading
        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=self.hidden_size,
            num_recurrent_layers=2,
            rnn_type='LSTM',
            backbone='resnet50',
        )
        self.actor_critic.load_state_dict(
            {
                k[len("actor_critic."):]: v
                for k, v in checkpoint['state_dict'].items()
                if "actor_critic" in k
            }
        )
        self.actor_critic.eval()

        self.hidden_state = torch.zeros(self.actor_critic.net.num_recurrent_layers, 1, checkpoint['model_args'].hidden_size)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long)

    def plan(self, depth, goal, t):
        batch = {
            'pointgoal_with_gps_compass': goal.view(1, -1),
            'depth': depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2]),
        }

        if t ==0:
            not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=depth.device)
        else:
            not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=depth.device)

        _, actions, _, self.hidden_state = self.actor_critic.act(
            batch,
            self.hidden_state.to(depth.device),
            self.prev_actions.to(depth.device),
            not_done_masks,
            deterministic=True,
        )
        self.prev_actions = torch.clone(actions)

        return actions.item()

    def reset(self):
        self.hidden_state = torch.zeros_like(self.hidden_state)
        self.prev_actions = torch.zeros_like(self.prev_actions)


class SemanticGrid(object):
    def __init__(self, batch_size, grid_dim, crop_size, cell_size, spatial_labels, object_labels, device):
        self.batch_size = batch_size
        self.grid_dim = grid_dim
        self.crop_size = crop_size
        self.cell_size = cell_size
        self.spatial_labels = spatial_labels
        self.object_labels = object_labels
        self.device = device

        self.crop_start = int((self.grid_dim[0] / 2) - (self.crop_size / 2))
        self.crop_end = int((self.grid_dim[0] / 2) + (self.crop_size / 2))

    # Transform each ground-projected grid into geocentric coordinates
    def spatialTransformer(self, grid, pose, abs_pose):
        geo_grid_out = torch.zeros(
            (grid.shape[0], grid.shape[1], self.grid_dim[0], self.grid_dim[1]),
            dtype=torch.float32,
        )

        init_pose = abs_pose[0, :]
        init_rot_mat = torch.tensor(
            [
                [torch.cos(init_pose[2]), -torch.sin(init_pose[2])],
                [torch.sin(init_pose[2]), torch.cos(init_pose[2])]
            ],
            dtype=torch.float32,
        )

        for i in range(grid.shape[0]):
            pose_step = pose[i, :]

            rel_coord = torch.tensor([pose_step[1], pose_step[0]], dtype=torch.float32)
            rel_coord = rel_coord.reshape((2, 1))
            rel_coord = torch.matmul(init_rot_mat, rel_coord)

            goal_grid_pos = torch.tensor([
                round(-rel_coord[1].item() / self.cell_size + 255),
                round(-rel_coord[0].item() / self.cell_size + 255),
            ])

        return geo_grid_out, goal_grid_pos

    # Transform a geocentric map back to egocentric view
    def rotate_map(self, grid, rel_pose, abs_pose):
        ego_grid_out = torch.zeros(
            (grid.shape[0], grid.shape[1], self.grid_dim[0], self.grid_dim[1]),
            dtype=torch.float32,
        ).to(grid.device)

        init_pose = abs_pose[0, :]
        init_rot_mat = torch.tensor(
            [
                [torch.cos(init_pose[2]), -torch.sin(init_pose[2])],
                [torch.sin(init_pose[2]), torch.cos(init_pose[2])]
            ],
            dtype=torch.float32,
        ).to(grid.device)

        for i in range(grid.shape[0]):
            rel_pose_step = rel_pose[i, :]

            rel_coord = torch.tensor([rel_pose_step[1], rel_pose_step[0]], dtype=torch.float32).to(grid.device)
            rel_coord = rel_coord.reshape((2, 1))
            rel_coord = torch.matmul(init_rot_mat, rel_coord)

            x = -2*(rel_coord[0] / self.cell_size) / (self.grid_dim[0])
            z = -2*(rel_coord[1] / self.cell_size) / (self.grid_dim[1])
            angle = -rel_pose_step[2]

            trans_theta = torch.tensor([[1, -0, x], [0, 1, z]], dtype=torch.float32).unsqueeze(0)
            rot_theta = torch.tensor(
                [
                    [torch.cos(angle), -torch.sin(angle), 0],
                    [torch.sin(angle), torch.cos(angle), 0]
                ],
                dtype=torch.float32,
            ).unsqueeze(0)
            trans_theta = trans_theta.to(grid.device)
            rot_theta = rot_theta.to(grid.device)

            grid_step = grid[i, :, :, :].unsqueeze(0)
            trans_disp_grid = F.affine_grid(trans_theta, grid_step.size(), align_corners=False)
            rot_disp_grid = F.affine_grid(rot_theta, grid_step.size(), align_corners=False)
            trans_ego_grid = F.grid_sample(grid_step, trans_disp_grid.float(), align_corners=False)
            ego_grid = F.grid_sample(trans_ego_grid, rot_disp_grid.float(), align_corners=False)
            ego_grid_out[i, :, :, :] = ego_grid

        return ego_grid_out


class utils():
    def get_rel_pose(self, pos2, pos1):
        x1, y1, o1 = pos1
        if len(pos2) == 2:  # if pos2 has no rotation
            x2, y2 = pos2
            dx = x2 - x1
            dy = y2 - y1
            return dx, dy
        else:
            x2, y2, o2 = pos2
            dx = x2 - x1
            dy = y2 - y1
            do = o2 - o1
            if do < -math.pi:
                do += 2 * math.pi
            if do > math.pi:
                do -= 2 * math.pi
            return dx, dy, do

    def discretize_coords(self, x, z, grid_dim, cell_size, translation=0):
        map_coords = torch.zeros((len(x), 2))
        xb = torch.floor(x[:]/cell_size) + (grid_dim[0]-1)/2.0
        zb = torch.floor(z[:]/cell_size) + (grid_dim[1]-1)/2.0 + translation
        xb = xb.int()
        zb = zb.int()
        map_coords[:,0] = xb
        map_coords[:,1] = zb
        # keep bin coords within dimensions
        map_coords[map_coords > grid_dim[0] - 1] = grid_dim[0] - 1
        map_coords[map_coords < 0] = 0
        return map_coords.long()

    def get_sim_location(self, agent_state):
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        height = agent_state.position[1]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        pose = x, y, o
        return pose, height

    def unravel_index(self, indices, shape):
        """Converts flat indices into unraveled coordinates in a target shape.
        This is a `torch` implementation of `numpy.unravel_index`.
        Args:
            indices: A tensor of indices, (*, N).
            shape: The targeted shape, (D,).
        Returns:
            unravel coordinates, (*, N, D).
        """
        shape = torch.tensor(shape)
        indices = indices % shape.prod()  # prevent out-of-bounds indices

        coord = torch.zeros(indices.size() + shape.size(), dtype=int)

        for i, dim in enumerate(reversed(shape)):
            coord[..., i] = indices % dim
            indices = indices // dim

        return coord.flip(-1)

    def get_coord_pose(self, sg, rel_pose, init_pose, grid_dim, cell_size, device=None):
        if isinstance(init_pose, list) or isinstance(init_pose, tuple):
            init_pose = torch.tensor(init_pose).unsqueeze(0)
        else:
            init_pose = init_pose.unsqueeze(0)

        zero_pose = torch.tensor([[0., 0., 0.]])
        if device != None:
            init_pose = init_pose.to(device)
            zero_pose = zero_pose.to(device)

        zero_coords = self.discretize_coords(
            x=zero_pose[:, 0],
            z=zero_pose[:, 1],
            grid_dim=(grid_dim, grid_dim),
            cell_size=cell_size,
        )

        pose_grid = torch.zeros((1, 1, grid_dim, grid_dim), dtype=torch.float32)
        pose_grid[0, 0, zero_coords[0,0], zero_coords[0,1]] = 1

        _, goal_grid_pos = sg.spatialTransformer(grid=pose_grid, pose=rel_pose, abs_pose=init_pose)
        inds = goal_grid_pos

        pose_coord = torch.zeros((1, 1, 2), dtype=torch.int64)
        pose_coord[0, 0, 0] = inds[1]
        pose_coord[0, 0, 1] = inds[0]
        return pose_coord

    def transform_ego_to_geo(self, ego_point, pose_coords, abs_pose_coords, abs_poses, t):
        rel_rot = torch.tensor(abs_poses[0][2]) - torch.tensor(abs_poses[t][2])
        dist_x = (ego_point[0, 0, 0] - pose_coords[0, 0, 0])
        dist_z = (ego_point[0, 0, 1] - pose_coords[0, 0, 1])
        rel_rot_mat = torch.tensor(
            [
                [torch.cos(rel_rot), -torch.sin(rel_rot)],
                [torch.sin(rel_rot), torch.cos(rel_rot)]
            ],
            dtype=torch.float32,
        )
        dist_vect = torch.tensor([dist_x, dist_z], dtype=torch.float)
        dist_vect = dist_vect.reshape((2, 1))
        rot_vect = torch.matmul(rel_rot_mat, dist_vect)

        abs_coords_x = abs_pose_coords[0, 0, 0] + rot_vect[0]
        abs_coords_z = abs_pose_coords[0, 0, 1] + rot_vect[1]
        abs_coords = torch.tensor([[[abs_coords_x, abs_coords_z]]])
        return abs_coords
