import os

from PIL import Image

import torch

from model_hub.llava.llava_hf.build_llava_hf import llava_hf_model_registry


class LLaVAHFHandler:
    def __init__(self, context):
        self.initialize(context)

    def initialize(self, context):
        self.device = torch.device(context.device)
        self.model = llava_hf_model_registry[context.model_id](
            model_id=context.model_id,
            device=context.device
        )

    def preprocess(self, data):
        filepath_image = data["filepath_image"]
        image0 = Image.open(os.path.join(filepath_image))
        data["image"] = image0
        data["pre_data"] = self.model.processor(
            images=data["image"], 
            text=f"USER: <image>\n{data["user_prompt"]}? ASSISTANT:",
            return_tensors="pt").to(device=self.device)
        return data

    def inference(self, data):
        data["inference"] = self.model.model.generate(
            **data["pre_data"],
            max_new_tokens=1024)
        return data

    def postprocess(self, data):
        generate_ids = self.model.processor.batch_decode(
            data["inference"], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        data["response"] = generate_ids[0].split("ASSISTANT: ")[-1].replace(".", "")
        return data
    
    def handle(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data
