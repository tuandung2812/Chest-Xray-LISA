import sys
sys.path.append(".")

import os
import torch
import deepspeed
from torch.utils.tensorboard import SummaryWriter

from training_hub.train_utils.parse_config import SetGlobalConfig
from training_hub.infrastructure.initialize_infra import initialize_infra
from dataloader_hub.lisa_dataloader.lisa_dataloader import prepare_dataset
from training_hub.train_utils.parse_arguments  import parse_arguments
from training_hub.train_lisa.trainer_lisa import LISATrainer
from model_hub.build_tokenizer import build_tokenizer
from model_hub.build_visual_encoder import build_visual_encoder
from model_hub.lisa.build_lisa import build_lisa 


def main():
    args = parse_arguments()
    config = SetGlobalConfig(args).cfg

    config = initialize_infra(config)
    tokenizer, config = build_tokenizer(config)  # update config with tokenizer
    visual_encoder = build_visual_encoder(config)
    lmm = build_lisa(
        config,
        tokenizer
    )
    dataset_train, dataset_val, collate_fn = prepare_dataset(
        config=config,
        visual_encoder=visual_encoder,
        tokenizer=tokenizer
    )
    steps_per_epoch = len(dataset_train) // config.train.batch_size
    config.train.steps_per_epoch = steps_per_epoch
    ds_config = {
        "train_micro_batch_size_per_gpu": config.train.batch_size,
        "gradient_accumulation_steps": config.model.optimizer.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.model.optimizer.lr,
                "weight_decay": 1e-4,
                "betas": (config.model.optimizer.beta1, config.model.optimizer.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": (config.train.epoch.end - config.train.epoch.start) * steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": config.model.optimizer.lr,
                "warmup_num_steps": 1000,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": config.model.lmm.precision == "fp16",
        },
        "bf16": {
            "enabled": config.model.lmm.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model, optimizer, dataloader_train, scheduler = deepspeed.initialize(
        model=lmm,
        model_parameters=lmm.parameters(),
        training_data=dataset_train,
        collate_fn=collate_fn,
        config=ds_config,
    )

    # resume deepspeed checkpoint
    # if args.auto_resume and len(args.resume) == 0:
    #     save_dir = config.train.dirpath_ckpt
    #     if os.path.exists(save_dir):
    #         args.resume = save_dir

    if args.resume:
        load_path, client_state = model.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_val, 
        shuffle=False, 
        drop_last=False
        )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=config.infrastructure.num_workers,
        pin_memory=False,
        sampler=val_sampler,
        collate_fn=collate_fn
    )
    writer = SummaryWriter(log_dir=config.train.dirpath_log)
    trainer = LISATrainer(
        config=config,
        dataloader_train=dataloader_train,
        dataloader_val=val_loader,
        tokenizer=tokenizer,
        model=model,
        scheduler=scheduler,
        writer=writer
    )
    trainer.train()


if __name__ == "__main__":
    main()
