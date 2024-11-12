from .modeling import LLaVAHF


def build_llava_hf(
    model_id,
    device
) -> LLaVAHF:
    model = LLaVAHF(
        model_id=model_id,
        device=device
    )
    return model


llava_hf_model_registry = {
    "llava-hf/llava-1.5-7b-hf": build_llava_hf,
} 