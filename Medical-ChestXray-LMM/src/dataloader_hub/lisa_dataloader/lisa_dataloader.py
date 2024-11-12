from functools import partial

from dataloader_hub.lisa_dataloader.lisa_ds_visual_question_answering.lisa_ds_vqa_vindr import LISAVisualQuestionAnsweringVinDrDataset
from dataloader_hub.lisa_dataloader.lisa_collate_fn import lisa_collate_fn


def prepare_dataset(
    config,
    visual_encoder,
    tokenizer,
):
    dataset_train = LISAVisualQuestionAnsweringVinDrDataset(
        config=config,
        dataset_split="train",
        visual_encoder=visual_encoder,
    )
    dataset_validation = LISAVisualQuestionAnsweringVinDrDataset(
        config=config,
        dataset_split="test",
        visual_encoder=visual_encoder,
    )
    collate_fn=partial(
        lisa_collate_fn,
        tokenizer=tokenizer,
        conv_type=config.dataset.conversation_template,
        use_mm_start_end=config.model.tokenizer.use_mm_start_end
        )
    return dataset_train, dataset_validation, collate_fn


def test():
    import deepspeed

    from training_hub.train_utils.parse_config import get_config
    from model_hub.build_tokenizer import build_tokenizer
    from model_hub.build_visual_encoder import build_visual_encoder
    from model_hub.lisa.build_lisa import build_lisa 

    config, _ = get_config(filepath_config="/mnt/12T/02_duong/LMM-Research-Foundation/src/training_hub/train_lisa/lisa_config/config_lisa_vqa_vindr.yaml")
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
    ds_config = {
        "train_micro_batch_size_per_gpu": config.train.batch_size,
        "gradient_accumulation_steps": config.model.optimizer.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.model.optimizer.lr,
                "weight_decay": 0.0,
                "betas": (config.model.optimizer.beta1, config.model.optimizer.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": (config.train.epoch.start - config.train.epoch.end) * config.train.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": config.model.optimizer.lr,
                "warmup_num_steps": 100,
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
    
    for data in dataloader_train:
        print(data)


if __name__ == "__main__":
    test()
