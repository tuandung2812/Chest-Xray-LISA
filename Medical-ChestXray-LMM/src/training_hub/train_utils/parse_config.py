import os
import yaml
import logging
from datetime import datetime
import importlib
from dotmap import DotMap

import torch

from training_hub.logger.logger import setup_logging


class SetGlobalConfig:
    def __init__(self, args):
        self.cfg, self.cfg_dict = get_config(args.filepath_config)
        dirpath_ckpt, dirpath_log = self.create_folders()
        self.cfg.train.dirpath_ckpt = dirpath_ckpt
        self.cfg.train.dirpath_log = dirpath_log

        self.log_levels = None
        self.set_log_level()
        self.cfg.logger = self.get_logger("trainer", self.cfg.train.verbosity)
        self.cfg.logger.info("Save folder created: {}".format(dirpath_ckpt))
        self.cfg.logger.info("Log folder created: {}".format(dirpath_log))

    def create_folders(self):
        version_name = self.cfg.version_name
        run_id = datetime.now().strftime(r"%d%m%Y_%H%M%S")
        date = run_id.split("_")[0]
        dirpath_ckpt = os.path.join(self.cfg.train.dirpath_ckpt, version_name, date, run_id, "model")
        dirpath_log = os.path.join(self.cfg.train.dirpath_ckpt, version_name, date, run_id, "log")
        filepath_config_dict = os.path.join(dirpath_ckpt, os.path.basename(self.cfg.filepath_config))
        os.makedirs(dirpath_ckpt, exist_ok=True)
        os.makedirs(dirpath_log, exist_ok=True)
        save_config(self.cfg_dict, filepath_config_dict)
        return dirpath_ckpt, dirpath_log

    def prepare_device(self, device_index):
        if device_index == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            device = torch.device("cpu")
            self.cfg.logger.info("Using ***** CPU\n")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_index
            device = torch.device("cuda")
            self.cfg.logger.info(f"Using ***** GPU {device_index}\n")
        return device

    def set_log_level(self):
        setup_logging(self.cfg.train.dirpath_log)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        
    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger


def save_config(config_dict, save_path):
    with open(save_path, "w") as f:
        yaml.dump(config_dict, f, indent=4, sort_keys=False)


def get_config(filepath_config):
    if not os.path.exists(filepath_config):
        raise FileNotFoundError("Could not find config file {}".format(filepath_config))
    with open(filepath_config, "r") as f:
        config_dict = yaml.safe_load(f)
        f.close()
    config = DotMap(config_dict)
    config.filepath_config = filepath_config
    return config, config_dict