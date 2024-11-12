from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration
)


class LLaVAHF:
    def __init__(
        self,
        model_id,
        device,
    ):
        self.processor = AutoProcessor.from_pretrained(
            model_id
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id
        ).to(device)
