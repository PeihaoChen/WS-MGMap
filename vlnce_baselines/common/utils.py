import os
import sys
import numpy as np
import zipfile
import socket
import shutil
import quaternion
from glob import glob
from shlex import quote
from typing import Dict, List

import torch


def transform_obs(
    observations: List[Dict], instruction_sensor_uuid: str, device=None
) -> Dict[str, torch.Tensor]:
    """Extracts instruction tokens from an instruction sensor and
    transposes a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        instruction_sensor_uuid: name of the instructoin sensor to
            extract from.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    for i in range(len(observations)):
        observations[i][instruction_sensor_uuid] = observations[i][
            instruction_sensor_uuid
        ]["tokens"]

    for obs in observations:
        if 'semantic' in obs:
            del obs['semantic']

    for obs in observations:
        for sensor in obs:
            if type(obs[sensor]) == torch.Tensor:
                obs[sensor] = obs[sensor].to(device)
    return observations


def check_exist_file(config):
    dirs = [config.VIDEO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
    if any([os.path.exists(d) for d in dirs]):
        if config.OVERWRITE:
            for d in dirs:
                if os.path.exists(d):
                    shutil.rmtree(d)
        else:
            order = None
            while order not in ['y', 'n']:
                order = input('Output directory already exists! Overwrite the folder? (y/n)')
                if order == 'y':
                    for d in dirs:
                        if os.path.exists(d):
                            shutil.rmtree(d)
                elif order == 'n':
                    break


def save_sh_n_codes(config, run_type, ignore_dir=['']):
    os.makedirs(config.CODE_DIR, exist_ok=True)

    name = os.path.join(config.CODE_DIR, 'run_{}_{}.sh'.format(run_type, socket.gethostname()))
    with open(name, 'w') as f:
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')

    name = os.path.join(config.CODE_DIR, 'code.zip')
    with zipfile.ZipFile(name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:

        first_list = glob('*', recursive=True)
        first_list = [i for i in first_list if i not in ignore_dir]

        file_list = []
        patterns = [x + '/**' for x in first_list]
        for pattern in patterns:
            file_list.extend(glob(pattern, recursive=True))

        file_list = [x[:-1] if x[-1] == "/" else x for x in file_list]
        for filename in file_list:
            zf.write(filename)


def save_config(config, run_type):
    os.makedirs(config.CONFIG_DIR, exist_ok=True)
    name = os.path.join(config.CONFIG_DIR, 'config_of_{}.txt'.format(run_type))
    with open(name, 'w') as f:
        f.write(str(config))


label_conversion_40_27 = {-1:0, 0:0, 1:15, 2:17, 3:1, 4:2, 5:3, 6:18, 7:19, 8:4, 9:15, 10:5, 11:6, 12:16, 13:20, 14:7, 15:8, 16:17, 17:17,
                    18:9, 19:21, 20:22, 21:16, 22:10, 23:11, 24:15, 25:12, 26:13, 27:23, 28:16, 29:16, 30:16, 31:16, 32:16,
                    33:24, 34:25, 35:16, 36:16, 37:14, 38:26, 39:16, 40:16}
label_conversion_40_3 = {-1:0, 0:0, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2,
                    18:1, 19:1, 20:1, 21:1, 22:1, 23:1, 24:1, 25:1, 26:1, 27:1, 28:1, 29:1, 30:1, 31:1, 32:1,
                    33:1, 34:1, 35:1, 36:1, 37:1, 38:1, 39:1, 40:1}


def get_sim_location(agent_state):
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

def load_scene_pcloud(preprocessed_scenes_dir, scene_id, n_object_classes):
    pcloud_path = preprocessed_scenes_dir+scene_id+'_pcloud.npz'
    if not os.path.exists(pcloud_path):
        raise Exception('Preprocessed point cloud for scene', scene_id,'not found!')

    data = np.load(pcloud_path)
    x = data['x']
    y = data['y']
    z = data['z']
    label_seq = data['label_seq']
    data.close()

    label_seq[ label_seq<0.0 ] = 0.0
    # Convert the labels to the reduced set of categories
    label_seq_spatial = label_seq.copy()
    label_seq_objects = label_seq.copy()
    for i in range(label_seq.shape[0]):
        curr_lbl = label_seq[i,0]
        label_seq_spatial[i] = label_conversion_40_3[curr_lbl]
        label_seq_objects[i] = label_conversion_40_27[curr_lbl]
    return (x, y, z), label_seq_spatial, label_seq_objects

def load_scene_color(preprocessed_scenes_dir, scene_id):
    # loads the rgb information of the map
    color_path = preprocessed_scenes_dir+scene_id+'_color.npz'
    if not os.path.exists(color_path):
        raise Exception('Preprocessed color for scene', scene_id,'not found!')

    data = np.load(color_path)
    r = data['r']
    g = data['g']
    b = data['b']
    color_pcloud = np.stack((r,g,b)) # 3 x Npoints
    return color_pcloud

def discretize_coords(x, z, grid_dim, cell_size, translation=0):
    # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
    # If translation=0, assumes the agent is at the center
    # If we want the agent to be positioned lower then use positive translation. When getting the gt_crop, we need negative translation
    #map_coords = torch.zeros((len(x), 2), device='cuda')
    map_coords = torch.zeros((len(x), 2))
    xb = torch.floor(x[:]/cell_size) + (grid_dim[0]-1)/2.0
    zb = torch.floor(z[:]/cell_size) + (grid_dim[1]-1)/2.0 + translation
    xb = xb.int()
    zb = zb.int()
    map_coords[:,0] = xb
    map_coords[:,1] = zb
    # keep bin coords within dimensions
    map_coords[map_coords>grid_dim[0]-1] = grid_dim[0]-1
    map_coords[map_coords<0] = 0
    return map_coords.long()

def slice_scene(x, y, z, label_seq, position, height, color_pcloud=None, device='cuda'):
    # z = -z
    # Slice the scene below and above the agent
    below_thresh = height-0.2
    above_thresh = height+2.0
    all_inds = np.arange(y.shape[0])
    below_inds = np.where(z<below_thresh)[0]
    above_inds = np.where(z>above_thresh)[0]
    # xout_inds = np.where(abs(x-position[1]) > 8)[0]
    # yout_inds = np.where(abs(y-position[0]) > 8)[0]
    invalid_inds = np.concatenate( (below_inds, above_inds), 0) # remove the floor and ceiling inds from the local3D points
    inds = np.delete(all_inds, invalid_inds)
    x_fil = x[inds]
    y_fil = y[inds]
    z_fil = z[inds]
    label_seq_fil = torch.tensor(label_seq[inds], dtype=torch.float, device=device)
    if color_pcloud is not None:
        color_pcloud_fil = torch.tensor(color_pcloud[:,inds], dtype=torch.float, device=device)
        return x_fil, y_fil, z_fil, label_seq_fil, color_pcloud_fil
    else:
        return x_fil, y_fil, z_fil, label_seq_fil

def get_gt_map(x, y, label_seq, abs_pose, grid_dim, cell_size, color_pcloud=None, z=None, device='cuda'):
    # Transform the ground-truth map to align with the agent's pose
    # The agent is at the center looking upwards
    point_map = np.array([x,y])
    angle = -abs_pose[2]
    rot_mat_abs = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    trans_mat_abs = np.array([[-abs_pose[1]],[abs_pose[0]]]) #### This is important, the first index is negative.
    ##rotating and translating point map points
    t_points = point_map - trans_mat_abs
    rot_points = np.matmul(rot_mat_abs,t_points)
    x_abs = torch.tensor(rot_points[0,:], device=device)
    y_abs = torch.tensor(rot_points[1,:], device=device)

    map_coords = discretize_coords(x=x_abs, z=y_abs, grid_dim=grid_dim, cell_size=cell_size)

    # Coordinates in map_coords need to be sorted based on their height, floor values go first
    # Still not perfect
    if z is not None:
        z = np.asarray(z)
        sort_inds = np.argsort(z)
        map_coords = map_coords[sort_inds,:]
        label_seq = label_seq[sort_inds,:]

    true_seg_grid = torch.zeros((grid_dim[0], grid_dim[1], 1), device=device)
    true_seg_grid[map_coords[:,1], map_coords[:,0]] = label_seq.clone()

    ### We need to flip the ground truth to align with the observations.
    ### Probably because the -y tp -z is a rotation about x axis which also flips the y coordinate for matteport.
    true_seg_grid = torch.flip(true_seg_grid, dims=[0])
    true_seg_grid = true_seg_grid.permute(2, 0, 1)

    if color_pcloud is not None:
        color_grid = torch.zeros((grid_dim[0], grid_dim[1], 3), device=device)
        color_grid[map_coords[:,1], map_coords[:,0],0] = color_pcloud[0]
        color_grid[map_coords[:,1], map_coords[:,0],1] = color_pcloud[1]
        color_grid[map_coords[:,1], map_coords[:,0],2] = color_pcloud[2]
        color_grid = torch.flip(color_grid, dims=[0])
        color_grid = color_grid.permute(2, 0 ,1)
        return true_seg_grid, color_grid/255.0
    else:
        return true_seg_grid


class TransfomationRealworldAgent():
    def __init__(self, agent_state) -> None:
        self.agent_state = agent_state
        self.T = self.agent_state.position.reshape(1,-1).T 
        self.R = quaternion.as_rotation_matrix(self.agent_state.rotation)

    def original_matrix(self, position):
        original_matrix = np.matrix(
            [[position[0]], [position[1]], [position[2]]]
        )
        return original_matrix

    def realworld2agent(self, point):
        O = self.original_matrix(point)
        point_a = (self.R.T @ O) + (self.R.T @ -self.T)
        return np.squeeze(np.asarray(point_a))

    def agent2realworld(self, point):
        O = self.original_matrix(point)
        point_w = (self.R @ O) + self.T
        return np.squeeze(np.asarray(point_w))
