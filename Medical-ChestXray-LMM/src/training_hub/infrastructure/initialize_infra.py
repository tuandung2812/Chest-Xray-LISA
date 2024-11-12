import torch


def initialize_infra(config):
    world_size = torch.cuda.device_count()
    config.train.distributed = world_size > 1
    return config
