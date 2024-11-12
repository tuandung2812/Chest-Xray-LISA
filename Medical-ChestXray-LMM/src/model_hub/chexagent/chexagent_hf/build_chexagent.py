from .modeling import CheXagentHF


def build_chexagent(
    model_id,
    dtype,
    device,
) -> CheXagentHF:
    model = CheXagentHF(
        model_id=model_id,
        dtype=dtype,
        device=device
    )
    return model


chexagent_model_registry = {
    "StanfordAIMI/CheXagent-8b": build_chexagent,
} 