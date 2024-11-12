import transformers

from dataloader_hub.lisa_dataloader.lisa_ds_config import *


def build_tokenizer(config):
    cfg_tokenizer = config.model.tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg_tokenizer.version,
        cache_dir=None,
        model_max_length=cfg_tokenizer.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    config.model.lmm.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if cfg_tokenizer.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    return tokenizer, config
