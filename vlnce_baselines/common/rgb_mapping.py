import math
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class Mapping(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.device = torch.device("cuda", model_config.gpu_id)
        self.num_proc = model_config.num_proc

        self.resolution = model_config.resolution
        self.egocentric_map_size = model_config.egocentric_map_size
        self.global_map_size = model_config.global_map_size
        self.global_map_depth = model_config.map_depth
        coordinate_min = - self.global_map_size * self.resolution / 2
        coordinate_max = self.global_map_size * self.resolution / 2

        self.to_grid = to_grid(self.global_map_size, coordinate_min, coordinate_max)
        self.rotate_tensor = RotateTensor(self.device)

        self.projection = Projection(self.egocentric_map_size, self.global_map_size, self.device, coordinate_min, coordinate_max)

        self.full_global_map = torch.zeros(self.num_proc, self.global_map_size, self.global_map_size, self.global_map_depth, device=self.device)
        self.agent_view = torch.zeros(self.num_proc, self.global_map_depth, self.global_map_size, self.global_map_size, device=self.device)

    def project_feat_to_map(self, features, full_global_map, observations, masks):
        bs = features.shape[0]
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        full_global_map[:bs, :, :, :] = full_global_map[:bs, :, :, :] * masks.unsqueeze(1).unsqueeze(1)

        proj_sem = self.projection.forward(features, observations['depth'] * 10, -(observations["compass"]))
        projection = torch.cat([proj_sem], dim=1)

        agent_view = self.agent_view[:bs] * 0
        agent_view[:, :, 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2): self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2): self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
        ] = projection
        st_pose = torch.cat(
            [
                -(grid_y.unsqueeze(1) - (self.global_map_size//2)) / (self.global_map_size//2),
                -(grid_x.unsqueeze(1) - (self.global_map_size//2)) / (self.global_map_size//2),
                torch.zeros_like(observations['compass']),
            ], dim=1
        )
        _, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
        translated = F.grid_sample(agent_view, trans_mat)

        fusion_map = torch.cat([full_global_map[:bs, :, :, :].unsqueeze(1), translated.permute(0, 2, 3, 1).unsqueeze(1)], dim=1)
        full_global_map[:bs, :, :, :], _ = torch.max(fusion_map, dim=1)
        st_pose_retrieval = torch.cat(
            [
                (grid_y.unsqueeze(1) - (self.global_map_size//2)) / (self.global_map_size//2),
                (grid_x.unsqueeze(1) - (self.global_map_size//2)) / (self.global_map_size//2),
                torch.zeros_like(observations['compass']),
            ], dim=1
        )
        _, trans_mat_retrieval = get_grid(st_pose_retrieval, agent_view.size(), self.device)
        translated_retrieval = F.grid_sample(full_global_map[:bs, :, :, :].permute(0, 3, 1, 2).contiguous(), trans_mat_retrieval)
        translated_retrieval = translated_retrieval[:, :,
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2): self.global_map_size//2 + math.ceil(self.egocentric_map_size/2),
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2): self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
        ]
        final_retrieval = self.rotate_tensor.forward(translated_retrieval, observations["compass"])

        return final_retrieval, full_global_map


class RGBMapping(Mapping):
    def __init__(self, model_config):
        super().__init__(model_config)

    def forward(self, rgb_features, observations, masks):
        if 'rgb_ego_map' not in observations:
            bs, c, h, w = rgb_features.shape
            rgb_features = rgb_features.permute(0, 2, 3, 1).reshape(bs, -1, c) # [bs, hxw, c]
            rgb_features = torch.nn.functional.adaptive_max_pool1d(rgb_features, self.global_map_depth) # [bs, hxw, self.global_map_depth]
            rgb_features = rgb_features.reshape(bs, h, w, -1).permute(0, 3, 1, 2)   # [bs, self.global_map_depth, h, w]
            final_retrieval, self.full_global_map = self.project_feat_to_map(rgb_features, self.full_global_map, observations, masks)
            observations['rgb_ego_map'] = final_retrieval
        else:
            final_retrieval = observations['rgb_ego_map']

        return final_retrieval


class to_grid():
    def __init__(self, global_map_size, coordinate_min, coordinate_max):
        self.global_map_size = global_map_size
        self.coordinate_min = coordinate_min
        self.coordinate_max = coordinate_max
        self.grid_size = (coordinate_max - coordinate_min) / global_map_size

    def get_grid_coords(self, positions):
        grid_x = ((self.coordinate_max - positions[:, 0]) / self.grid_size).round()
        grid_y = ((positions[:, 1] - self.coordinate_min) / self.grid_size).round()
        return grid_x, grid_y


def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


class ComputeSpatialLocs():
    def __init__(self, egocentric_map_size, global_map_size, device, coordinate_min, coordinate_max):
        self.device = device
        self.egocentric_map_size = egocentric_map_size
        self.local_scale = float(coordinate_max - coordinate_min) / float(global_map_size)

    def get_camera_matrix(self, imh, imw, fov):
        self.cx, self.cy = imh / 2., imw / 2.
        self.fx = (imh / 2.) / np.tan(np.deg2rad(fov / 2.))
        self.fy = (imw / 2.) / np.tan(np.deg2rad(fov / 2.))

    def forward(self, depth):
        depth = depth.permute(0, 3, 1, 2)
        _, _, imh, imw = depth.shape  # batchsize, 1, imh, imw

        self.get_camera_matrix(imh, imw, 90)

        x = rearrange(torch.arange(0, imw), 'w -> () () () w').to(self.device)
        y = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()').to(self.device)
        xx = (x - self.cx) / self.fx
        yy = (y - self.cy) / self.fy

        # 3D real-world coordinates (in meters)
        Z = depth
        X = xx * Z
        Y = yy * Z

        # Valid inputs
        valid_inputs = (depth != 0) & ((Y > -1.5) & (Y < 0.1))

        # X ground projection and Y ground projection
        x_gp = ((X / self.local_scale) + (self.egocentric_map_size - 1) / 2).round().long()  # (bs, imh, imw, 1)
        y_gp = (-(Z / self.local_scale) + (self.egocentric_map_size - 1) / 2).round().long()  # (bs, imh, imw, 1)

        return torch.cat([x_gp, y_gp], dim=1), valid_inputs


class ProjectToGroundPlane():
    def __init__(self, egocentric_map_size, device):
        self.egocentric_map_size = egocentric_map_size
        self.device = device

    def forward(self, conv, spatial_locs, valid_inputs):
        outh, outw = (self.egocentric_map_size, self.egocentric_map_size)
        bs, f, HbyK, WbyK = conv.shape
        eps = -1e16
        depth_h = spatial_locs.shape[-1]
        K = depth_h / WbyK  # Hardcoded value of K

        # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
        idxes_ss = ((torch.arange(0, HbyK, 1) * K).long().to(self.device), \
                    (torch.arange(0, WbyK, 1) * K).long().to(self.device))

        spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]]  # (bs, 2, HbyK, WbyK)
        valid_inputs_ss = valid_inputs[:, :, idxes_ss[0][:, None], idxes_ss[1]]  # (bs, 1, HbyK, WbyK)
        valid_inputs_ss = valid_inputs_ss.squeeze(1)  # (bs, HbyK, WbyK)
        invalid_inputs_ss = ~valid_inputs_ss

        # Filter out invalid spatial locations
        invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0) | \
                               (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0)  # (bs, H, W)

        invalid_writes = invalid_spatial_locs | invalid_inputs_ss

        # Set the idxes for all invalid locations to (0, 0)
        spatial_locs_ss[:, 0][invalid_writes] = 0
        spatial_locs_ss[:, 1][invalid_writes] = 0

        # Weird hack to account for max-pooling negative feature values
        invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').float()
        conv_masked = conv * (1 - invalid_writes_f) + eps * invalid_writes_f
        conv_masked = rearrange(conv_masked, 'b e h w -> b e (h w)')

        # Linearize ground-plane indices (linear idx = y * W + x)
        linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0]  # (bs, H, W)
        linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
        linear_locs_ss = linear_locs_ss.expand(-1, f, -1)  # .contiguous()

        proj_feats, _ = torch_scatter.scatter_max(
            conv_masked,
            linear_locs_ss,
            dim=2,
            dim_size=outh * outw,
        )
        proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)

        # Replace invalid features with zeros
        eps_mask = (proj_feats == eps).float()
        proj_feats = proj_feats * (1 - eps_mask) + eps_mask * (proj_feats - eps)

        return proj_feats


class RotateTensor:
    def __init__(self, device):
        self.device = device

    def forward(self, x_gp, heading):
        sin_t = torch.sin(heading.squeeze(1))
        cos_t = torch.cos(heading.squeeze(1))
        A = torch.zeros(x_gp.size(0), 2, 3).to(self.device)
        A[:, 0, 0] = cos_t
        A[:, 0, 1] = sin_t
        A[:, 1, 0] = -sin_t
        A[:, 1, 1] = cos_t

        grid = F.affine_grid(A, x_gp.size())
        rotated_x_gp = F.grid_sample(x_gp, grid)
        return rotated_x_gp


class Projection:
    def __init__(self, egocentric_map_size, global_map_size, device, coordinate_min, coordinate_max):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.compute_spatial_locs = ComputeSpatialLocs(
            egocentric_map_size, global_map_size,
            device, coordinate_min, coordinate_max
        )
        self.project_to_ground_plane = ProjectToGroundPlane(egocentric_map_size, device)
        self.rotate_tensor = RotateTensor(device)

    def forward(self, conv, depth, heading):
        spatial_locs, valid_inputs = self.compute_spatial_locs.forward(depth)
        x_gp = self.project_to_ground_plane.forward(conv, spatial_locs, valid_inputs)
        rotated_x_gp = self.rotate_tensor.forward(x_gp, heading)
        return rotated_x_gp
