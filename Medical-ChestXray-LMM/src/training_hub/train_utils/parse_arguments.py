import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--filepath-config", default="/mnt/12T/02_duong/LMM-Research-Foundation/src/training_hub/train_lisa/lisa_config/config_lisa_vqa_vindr.yaml", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    args = parser.parse_args()
    return args

