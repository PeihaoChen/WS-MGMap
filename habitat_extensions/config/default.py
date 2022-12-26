from typing import List, Optional, Union

from habitat.config.default import Config as CN
from habitat.config.default import get_config


_C = get_config()
_C.defrost()

# -----------------------------------------------------------------------------
# VLN ORACLE ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_ACTION_SENSOR = CN()
_C.TASK.VLN_ORACLE_ACTION_SENSOR.TYPE = "VLNOracleActionSensor"
_C.TASK.VLN_ORACLE_ACTION_SENSOR.GOAL_RADIUS = 0.5
# compatibility with the dataset generation oracle and paper results.
# if False, use the ShortestPathFollower in Habitat
_C.TASK.VLN_ORACLE_ACTION_SENSOR.USE_ORIGINAL_FOLLOWER = True
# -----------------------------------------------------------------------------
# VLN ORACLE PROGRESS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR = CN()
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR.TYPE = "VLNOracleProgressSensor"
# -----------------------------------------------------------------------------
# VLN ORACLE WAYPOINT SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR = CN()
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.TYPE = "VLNOracleWaypointSensor"
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.GOAL_RADIUS = 0.5
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.USE_ORIGINAL_FOLLOWER = True
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.MAP_SIZE = 100
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.MAP_RESOLUTION = 1250
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.LAW = CN()
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.LAW.USE = True
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.LAW.SPLIT = "train"
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.LAW.GT_PATH = "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz"
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.LAW.IS_SPARSE = True
_C.TASK.VLN_ORACLE_WAYPOINT_SENSOR.LAW.NUM_WAYPOINTS = 6
# -----------------------------------------------------------------------------
# VLN ORACLE PATH SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_PATH_SENSOR = CN()
_C.TASK.VLN_ORACLE_PATH_SENSOR.TYPE = "VLNOraclePathSensor"
_C.TASK.VLN_ORACLE_PATH_SENSOR.MAP_RESOLUTION = 1250
_C.TASK.VLN_ORACLE_PATH_SENSOR.MAP_SIZE = 100
_C.TASK.VLN_ORACLE_PATH_SENSOR.LINE_WIDTH = 1
# -----------------------------------------------------------------------------
# SEMANTIC FILTER SENSOR
# -----------------------------------------------------------------------------
_C.TASK.SEMANTIC_FILTER_SENSOR = CN()
_C.TASK.SEMANTIC_FILTER_SENSOR.TYPE = "SemanticFilterSensor"
_C.TASK.SEMANTIC_FILTER_SENSOR.HEIGHT = 256
_C.TASK.SEMANTIC_FILTER_SENSOR.WIDTH = 256
_C.TASK.SEMANTIC_FILTER_SENSOR.CATEGORY = 27
# -----------------------------------------------------------------------------
# GT SEMANTIC MAP SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GT_SEMANTIC_MAP_SENSOR = CN()
_C.TASK.GT_SEMANTIC_MAP_SENSOR.TYPE = "GtSemanticMapSensor"
_C.TASK.GT_SEMANTIC_MAP_SENSOR.MAP_SIZE = 100
_C.TASK.GT_SEMANTIC_MAP_SENSOR.SPLIT = 'train'  # 'train', 'train_aug'
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"


# -----------------------------------------------------------------------------
# NDTW MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.NDTW = CN()
_C.TASK.NDTW.TYPE = "NDTW"
_C.TASK.NDTW.SPLIT = "val_seen"
_C.TASK.NDTW.FDTW = True  # False: DTW
_C.TASK.NDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.NDTW.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# SDTW MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SDTW = CN()
_C.TASK.SDTW.TYPE = "SDTW"
_C.TASK.SDTW.SPLIT = "val_seen"
_C.TASK.SDTW.FDTW = True  # False: DTW
_C.TASK.SDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.SDTW.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# PATH_LENGTH MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PATH_LENGTH = CN()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"
# -----------------------------------------------------------------------------
# ORACLE_NAVIGATION_ERROR MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_NAVIGATION_ERROR = CN()
_C.TASK.ORACLE_NAVIGATION_ERROR.TYPE = "OracleNavigationError"
# -----------------------------------------------------------------------------
# ORACLE_SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SUCCESS = CN()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# ORACLE_SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SPL = CN()
_C.TASK.ORACLE_SPL.TYPE = "OracleSPL"
_C.TASK.ORACLE_SPL.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# STEPS_TAKEN MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.STEPS_TAKEN = CN()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"

_C.DATASET.split_num = 0
_C.DATASET.split_rank = 0


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None, opts: Optional[list] = None
) -> CN:
    """Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)
    config.freeze()
    return config
