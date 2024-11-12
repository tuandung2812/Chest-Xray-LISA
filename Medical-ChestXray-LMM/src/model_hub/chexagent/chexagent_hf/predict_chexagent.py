import os

from PIL import Image
import torch

from model_hub.chexagent.build_chexagent import chexagent_model_registry


class Chexagent_Handler:
    def __init__(self, context):
        self.initialize(context)

    def initialize(self, context):
        self.device = torch.device(context.device)
        self.dtype = torch.float16
        self.model = chexagent_model_registry[context.model_id](
            model_id=context.model_id,
            dtype=self.dtype,
            device=self.device,
        )

    def handle(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data

    def preprocess(self, data):
        filepath_image = data["filepath_image"]
        image0 = Image.open(os.path.join(filepath_image))
        data["image"] = image0
        data["pre_data"] = self.model.processor(
            images=data["image"], 
            text=f" USER: <s>{data["user_prompt"]} ASSISTANT: <s>", 
            return_tensors="pt").to(device=self.device, dtype=self.dtype)
        return data

    def inference(self, data):
        data["inference"] = self.model.model.generate(
            **data["pre_data"],
            generation_config=self.model.generation_config)[0]
        return data

    def postprocess(self, data):
        data["response"] = self.model.processor.tokenizer.decode(
            data["inference"], 
            skip_special_tokens=True
        )
        return data
