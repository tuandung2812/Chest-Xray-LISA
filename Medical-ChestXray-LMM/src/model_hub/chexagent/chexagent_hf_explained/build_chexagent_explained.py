from .modeling import CheXagentHFExplained


def build_chexagent_explained(
    model_id,
    dtype,
    device,
) -> CheXagentHFExplained:
    model = CheXagentHFExplained(
        model_id=model_id,
        dtype=dtype,
        device=device
    )
    return model


chexagent_explained_model_registry = {
    "StanfordAIMI/CheXagent-8b": build_chexagent_explained,
} 