import math
from typing import Dict

import numpy as np
import torch

from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision


cv2 = try_cv2_import()

COLOR_ProjSem_27 = [
[255,255,255] # white
,[128,128,0] # olive (dark yellow)
,[0,0,255] # blue
,[255,0,0] # red
,[255,0,255] # magenta
,[0,255,255] # cyan
,[255,165,0] # orange
,[255,255,0] # yellow
,[128,128,128] # gray
,[128,0,0] # maroon
,[255,20,147] # pink 
,[0,128,0] # dark green
,[128,0,128] # purple
,[0,128,128] # teal
,[0,0,128] # navy (dark blue)
,[210,105,30] # chocolate
,[188,143,143] # rosy brown
,[0,255,0] # green
,[255,215,0] # gold
,[0,0,0] # black
,[192,192,192] # silver
,[138,43,226] # blue violet
,[255,127,80] # coral
,[238,130,238] # violet
,[245,245,220] # beige
,[139,69,19] # saddle brown
,[64,224,208] # turquoise
]

OBJECTS_ProjSem_27 = [
    'void', 'chair', 'door', 'table', 'cushion',
    'sofa', 'bed', 'plant', 'sink', 'toilet', 
    'tv_monitor', 'shower', 'bathtub',
    'counter', 'appliances', 'structure', 'other',
    'free-space', 'picture', 'cabinet', 'chest_of_drawers', 'stool',
    'towel', 'fireplace', 'gym_equipment', 'seating',
    'clothes'
]

COLOR_ProjSem = [
    [235, 190, 157], [235, 219, 156], [255, 255, 255], [189, 234, 155], [163, 233, 158],
    [156, 234, 180], [156, 235, 206], [157, 226, 236], [156, 198, 235], [156, 170, 231],
    [170, 155, 235], [198, 154, 234], [230, 156, 235], [234, 154, 213], [235, 156, 181],
    [157, 190, 181], [198, 156, 206],
]
OBJECTS_ProjSem = [
    'wall', 'chair', 'door', 'table', 'picture',
    'cabinet', 'window', 'sofa', 'bed', 'plant',
    'sink', 'stairs', 'mirror', 'shower', 'counter',
    'fireplace', 'railing',
]

COLOR_AABBSem = [
    [235, 190, 157], [235, 219, 156], [208, 234, 157], [189, 234, 155], [163, 233, 158],
    [156, 234, 180], [156, 235, 206], [157, 226, 236], [156, 198, 235], [156, 170, 231],
    [170, 155, 235], [198, 154, 234], [230, 156, 235], [234, 154, 213], [235, 156, 181],
]
OBJECTS_AABBSem = [
    'door', 'stair', 'bed', 'doorway', 'table',
    'chair', 'couch', 'sink', 'closet', 'fireplace',
    'rug', 'counter', 'desk', 'painting', 'window',
]

COCO_COLOR = [
    [1.0, 1.0, 1.0],
    [0.6, 0.6, 0.6],
    [0.95, 0.95, 0.95],
    [0.96, 0.36, 0.26],
    [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
    [0.9400000000000001, 0.7818, 0.66],
    [0.9400000000000001, 0.8868, 0.66],
    [0.8882000000000001, 0.9400000000000001, 0.66],
    [0.7832000000000001, 0.9400000000000001, 0.66],
    [0.6782000000000001, 0.9400000000000001, 0.66],
    [0.66, 0.9400000000000001, 0.7468000000000001],
    [0.66, 0.9400000000000001, 0.8518000000000001],
    [0.66, 0.9232, 0.9400000000000001],
    [0.66, 0.8182, 0.9400000000000001],
    [0.66, 0.7132, 0.9400000000000001],
    [0.7117999999999999, 0.66, 0.9400000000000001],
    [0.8168, 0.66, 0.9400000000000001],
    [0.9218, 0.66, 0.9400000000000001],
    [0.9400000000000001, 0.66, 0.8531999999999998],
    [0.9400000000000001, 0.66, 0.748199999999999]]

COCO_OBJECTS = [ 'unexplored', 'obstacle', 'free', 'waypoint', 'agent',
    'chair', 'couch', 'potted plant', 'bed', 'toilet',
    'tv', 'dining-table', 'oven', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'cup', 'bottle',
]

COLOR_HEAT = {
    'R': [
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
    ],
    'G': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ],
    'B': [
        0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
}


def observations_to_image(observation: Dict, info: Dict, waypoint_info, att_map) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    # if "depth" in observation:
    #     if observation_size == -1:
    #         observation_size = observation["depth"].shape[0]
    #     depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
    #     depth_map = np.stack([depth_map for _ in range(3)], axis=2)
    #     depth_map = cv2.resize(
    #         depth_map,
    #         dsize=(observation_size, observation_size),
    #         interpolation=cv2.INTER_CUBIC,
    #     )
    #     egocentric_view.append(depth_map)

    assert len(egocentric_view) > 0, "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info:
        top_down_map = info["top_down_map"]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info["top_down_map"]["fog_of_war_mask"]
        )
        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=top_down_map.shape[0] // 16,
        )

        # if 'waypoint' in waypoint_info:
        #     waypoint = maps.to_grid(
        #         waypoint_info['action'][0],
        #         waypoint_info['action'][2],
        #         maps.COORDINATE_MIN, maps.COORDINATE_MAX, (1250, 1250),
        #     )
        #     crop_waypoint = (
        #         waypoint[0] - info['top_down_map']['waypoint']['ind_x_min'],
        #         waypoint[1] - info['top_down_map']['waypoint']['ind_y_min']
        #     )
        #     maps.draw_path(
        #         top_down_map=top_down_map,
        #         path_points=[crop_waypoint, crop_waypoint],
        #         color=[200, 0, 0],
        #         thickness=5,
        #     )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((frame, top_down_map), axis=1)

        # DRAW SEMANTIC MAP
        ego_map = observation['ego_map_vis']
        channel = ego_map.shape[0]
        semantic_map = np.ones([*ego_map.shape[1:], 3], dtype=np.uint8) * 255
        
        if channel == 17: # AABBSem: occ map + history path + 15 objects
            offset = 2
            objects, color = OBJECTS_AABBSem, COLOR_AABBSem
            semantic_map[ego_map[0, :, :] == 1, :] = [75, 75, 75]
        elif channel == 18: # PorjSem: occ map + 17 objects
            offset = 1
            objects, color = OBJECTS_ProjSem, COLOR_ProjSem
            semantic_map[ego_map[0, :, :] < 0.1, :] = [75, 75, 75]
        elif channel == 29: # ProjSem: occ map + explored map + 27 objects
            offset = 2
            objects, color = OBJECTS_ProjSem_27, COLOR_ProjSem_27

        for i in range(len(objects)):
            semantic_map[ego_map[i + offset, :, :] > 0.5, :] = color[i]

        semantic_map = maps.draw_agent(
            image=semantic_map,
            agent_center_coord=[50, 50], # FIXME 用参数代替
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=top_down_map.shape[0] // 64,
        )
        wp_grid_x = -torch.tanh(waypoint_info['action'])[1] * 50 + 50
        wp_grid_y = torch.tanh(waypoint_info['action'])[0] * 50 + 50
        _limit = lambda x: min(max(int(x), 0), 100)
        semantic_map[_limit(wp_grid_x - 2):_limit(wp_grid_x + 2),
                     _limit(wp_grid_y - 2):_limit(wp_grid_y + 2), :] = [200, 0, 0]  # draw waypoint

        semantic_map = cv2.resize(semantic_map,
                                  (observation_size, observation_size),
                                  interpolation=cv2.INTER_CUBIC)
        frame = np.concatenate((frame, semantic_map), axis=1)

        legend = np.ones([observation_size, 120, 3], dtype=np.uint8) * 255
        grid = legend.shape[0] // 30 * 2
        for i in range(len(objects)):
            cv2.rectangle(legend, (grid, grid * i + 10), (grid * 2, grid * i + 10 + grid // 2), color[i], -1)
            cv2.putText(legend, objects[i], (grid * 2 + 5, grid * i + 10 + grid // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        frame = np.concatenate((frame, legend), axis=1)

        vis_att_map = np.ones([att_map.shape[0], 3], dtype=np.uint8) * 255
        for idx, value in enumerate(att_map):
            color_idx = ((1 - (value - att_map.min()) / (att_map.max() - att_map.min() + 1e-6)) * (len(COLOR_HEAT['R'])) - 1)
            color_idx = int(color_idx.item())
            vis_att_map[idx, :] = [COLOR_HEAT['R'][color_idx], COLOR_HEAT['G'][color_idx], COLOR_HEAT['B'][color_idx]]
        vis_att_map = vis_att_map.reshape(24, 24, 3)
        vis_att_map = cv2.resize(vis_att_map, (observation_size, observation_size), interpolation=cv2.INTER_CUBIC)
        frame = np.concatenate((frame, vis_att_map), axis=1)

    return frame
