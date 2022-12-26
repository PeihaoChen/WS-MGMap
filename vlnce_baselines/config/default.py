import os
from typing import List, Optional, Union

from habitat.config.default import Config as CN
from habitat_extensions.config.default import get_extended_config as get_task_config


# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_ID = 0
_C.SIMULATOR_GPU_IDS = None
_C.TORCH_GPU_ID = 0
_C.NUM_PROCESSES = 4
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.LOG_FILE = "train.log"
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_UPDATES = 300000
_C.CHECKPOINT_INTERVAL = 512000

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.SPLIT = "val_seen"  # The split to evaluate on
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.EPISODE_COUNT = 2

# -----------------------------------------------------------------------------
# INFERENCE CONFIG
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"

# -----------------------------------------------------------------------------
# DAGGER ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.DAGGER = CN()
_C.DAGGER.LR = 2.5e-4
_C.DAGGER.ITERATIONS = 10
_C.DAGGER.EPOCHS = 4
_C.DAGGER.UPDATE_SIZE = 5000
_C.DAGGER.BATCH_SIZE = 5
_C.DAGGER.P = 0.75
_C.DAGGER.LMDB_MAP_SIZE = 5.0e12
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens/
_C.DAGGER.LMDB_COMMIT_FREQUENCY = 50
_C.DAGGER.USE_IW = True
# If True, load precomputed features directly from LMDB_FEATURES_DIR.
_C.DAGGER.PRELOAD_LMDB_FEATURES = False
_C.DAGGER.LMDB_FEATURES_DIR = "data/trajectories_dirs/debug/trajectories.lmdb"
# load an already trained model for fine tuning
_C.DAGGER.LOAD_FROM_CKPT = False
_C.DAGGER.CKPT_TO_LOAD = "data/checkpoints/ckpt.0.pth"

# -----------------------------------------------------------------------------
# MODELING CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# on GT trajectories in the training set
_C.MODEL.inflection_weight_coef = 3.2

_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.max_length = 200
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = False
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = True
_C.MODEL.INSTRUCTION_ENCODER.backbone = 'lstm'

_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.output_size = 256
_C.MODEL.RGB_ENCODER.backbone = "unet"
_C.MODEL.RGB_ENCODER.pretrain_model = 'data/pretrain_model/unet-models/2021_02_14-23_42_50.pt'

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.output_size = 128
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"  # type of resnet to use
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = "data/pretrain_model/ddppo-models/gibson-2plus-resnet50.pth"  # path to DDPPO resnet weights

_C.MODEL.MAP_ENCODER = CN()
_C.MODEL.MAP_ENCODER.ego_map_size = 100
_C.MODEL.MAP_ENCODER.output_size = 256

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"
_C.MODEL.STATE_ENCODER.input_type = ['rgb', 'depth', 'map']

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = True
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier

_C.MODEL.CONTRASTIVE_MONITOR = CN()
_C.MODEL.CONTRASTIVE_MONITOR.target_tau = 0.07
_C.MODEL.CONTRASTIVE_MONITOR.use = True
_C.MODEL.CONTRASTIVE_MONITOR.alpha = 1.0

_C.MODEL.PREDICTION_MONITOR = CN()
_C.MODEL.PREDICTION_MONITOR.use = True
_C.MODEL.PREDICTION_MONITOR.alpha = 0.1

_C.MODEL.RGBMAPPING = CN()
_C.MODEL.RGBMAPPING.map_depth = 64
_C.MODEL.RGBMAPPING.global_map_size = 240
_C.MODEL.RGBMAPPING.egocentric_map_size = 100
_C.MODEL.RGBMAPPING.resolution = 0.12
_C.MODEL.RGBMAPPING.gpu_id = 0
_C.MODEL.RGBMAPPING.num_proc = 1

_C.STOP_CONDITION = CN()
_C.STOP_CONDITION.TYPE = 'prog'
_C.STOP_CONDITION.PROG_THRESHOLD = 0.8

_C.OVERWRITE = False
_C.LOG_INTERVAL = 100
_C.random_agent = False
_C.RESUME_CKPT = None # resume from this ckpt
_C.VIDEO_NUM = 99999
_C.ego_map_size = 100
_C.same_level_train = False
_C.ep_max_len = 200
_C.step_num = 3
_C.use_ddppo = False


def get_config(
    config_paths: Optional[Union[List[str], str]] = None, opts: Optional[list] = None
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if config.BASE_TASK_CONFIG_PATH != "":
        config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    return config


def refine_config(config, local_rank):
    config.defrost()

    config.TORCH_GPU_ID = local_rank
    config.MODEL.RGBMAPPING.gpu_id = config.TORCH_GPU_ID
    config.MODEL.RGBMAPPING.num_proc = config.NUM_PROCESSES

    split = config.TASK_CONFIG.DATASET.SPLIT
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.TASK_CONFIG.TASK.VLN_ORACLE_WAYPOINT_SENSOR.LAW.SPLIT = split

    if config.DAGGER.P == 1.0: # if doing teacher forcing, don't switch the scene until it is complete
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (-1)

    if config.same_level_train:
        config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train_same_level.json.gz'

    if 'aug' in config.BASE_TASK_CONFIG_PATH:
        config.TASK_CONFIG.TASK.GT_SEMANTIC_MAP_SENSOR.SPLIT = 'train_aug'

    config.freeze()
    return config

def set_saveDir_GPUs(config, run_type, model_dir, note, gpus, local_rank):
    config.defrost()

    run_dir = os.path.join(model_dir, "run_{}_{}".format(run_type, note))
    os.makedirs(run_dir, exist_ok=True)

    config.CHECKPOINT_FOLDER = os.path.join(run_dir, 'checkpoint')
    config.LOG_FILE = os.path.join(run_dir, '{}.log'.format(run_type))
    config.TENSORBOARD_DIR = os.path.join(run_dir, 'tensorboard')
    if config.DAGGER.PRELOAD_LMDB_FEATURES is False:
        config.DAGGER.LMDB_FEATURES_DIR = os.path.join(run_dir, 'trajectories.lmdb')
    config.VIDEO_DIR = os.path.join(run_dir, 'video_dir')
    config.CODE_DIR = os.path.join(run_dir, 'sh_n_codes')
    config.CONFIG_DIR = os.path.join(run_dir, 'config')
    config.METRIC_DIR = os.path.join(run_dir, 'metric')

    config.SIMULATOR_GPU_ID = local_rank
    config.SIMULATOR_GPU_IDS = None
    if gpus is not None:
        config.TORCH_GPU_ID = gpus[0]
        config.SIMULATOR_GPU_IDS = gpus if len(gpus) == 1 else gpus[1:]
    config.freeze()

    return config
