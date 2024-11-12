from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    GenerationConfig
)


class CheXagentHF:
    def __init__(
        self,
        model_id,
        dtype,
        device
    ):
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            trust_remote_code=True
        ).to(device)
        self.generation_config = GenerationConfig.from_pretrained(
            model_id,
        )
