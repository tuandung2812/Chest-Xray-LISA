from pathlib import Path
import logging
import logging.config

from training_hub.train_utils.read_files import read_json


def setup_logging(save_dir, log_config="training_hub/logger/logger_config.json", default_level=logging.INFO):
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        save_dir = Path(save_dir)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

            logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
