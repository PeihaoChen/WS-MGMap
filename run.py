#!/usr/bin/env python3

import argparse
import random
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import List
plt.switch_backend('agg')
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
warnings.filterwarnings("ignore")

import torch

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.config.default import get_config, refine_config, set_saveDir_GPUs
from vlnce_baselines.common.utils import check_exist_file, save_sh_n_codes, save_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        default="train",
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "-c", "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "-e", "--model-dir",
        default=None,
        help="path to save checkpoint, log and others",
    )
    parser.add_argument(
        "--note",
        default='base',
        help="add extra note for running file",
    )
    parser.add_argument(
        "-g", "--gpus",
        default=None,
        nargs="+",
        type=int,
        help="GPU id to run experiments",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training'
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str,
            run_type: str,
            model_dir: str,
            note: str,
            gpus: List[int],
            opts=None,
            local_rank=-1) -> None:
    """Runs experiment given mode and config
    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        model_dir: path to save.
        note: extra note.
        opts: list of strings of additional config options.
    Returns:
        None.
    
    """
    config = get_config(exp_config, opts)
    config = set_saveDir_GPUs(config, run_type, model_dir, note, gpus, local_rank)
    config = refine_config(config, local_rank)
    if local_rank == 0:
        check_exist_file(config)
        save_sh_n_codes(
            config,
            run_type,
            ignore_dir=['habitat-lab', 'data', 'result', 'habitat-sim', 'temp']
        )
        save_config(config, run_type)
    logger.add_filehandler(config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()
