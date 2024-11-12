import torch
from peft import LoraConfig, get_peft_model

from model_hub.lisa.lisa_0.LISA import LISAForCausalLM


def build_lisa(
    config,
    tokenizer,
):
    cfg_lmm = config.model.lmm
    model_args = {
        "train_mask_decoder": cfg_lmm.train_mask_decoder,
        "out_dim": cfg_lmm.out_dim,
        "ce_loss_weight": cfg_lmm.ce_loss_weight,
        "dice_loss_weight": cfg_lmm.dice_loss_weight,
        "bce_loss_weight": cfg_lmm.bce_loss_weight,
        "seg_token_idx": cfg_lmm.seg_token_idx,
        "vision_pretrained": cfg_lmm.vision_pretrained,
        "vision_tower": cfg_lmm.vision_tower,
        "use_mm_start_end": cfg_lmm.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if cfg_lmm.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg_lmm.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        cfg_lmm.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=config.infrastructure.local_rank)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    lora_r = cfg_lmm.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = cfg_lmm.lora_alpha
        lora_dropout = cfg_lmm.lora_dropout
        lora_target_modules = find_linear_layers(
            model, cfg_lmm.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    return model
