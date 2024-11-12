from transformers import GenerationConfig

from .processing_chexagent import CheXagentProcessor
from .modeling_chexagent import CheXagentForConditionalGeneration


class CheXagentHFExplained:
    def __init__(
        self,
        model_id,
        dtype,
        device
    ):
        self.processor = CheXagentProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        self.model = CheXagentForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            trust_remote_code=True
        ).to(device)
        self.generation_config = GenerationConfig.from_pretrained(
            model_id,
        )