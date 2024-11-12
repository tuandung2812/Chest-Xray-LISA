import sys
sys.path.append(".")

import os
from dotmap import DotMap
from tqdm import tqdm
import json

import transformers

from evaluation_hub.load_data.load_vindr_data import load_vindr_data
from model_hub.llava.llava_hf.predict_llava_hf import LLaVAHFHandler

transformers.logging.set_verbosity_error()


def evaluate_llavahf(
    filepath_metadata,
    dirpath_images,
    dirpath_output,
):  
    os.makedirs(dirpath_output, exist_ok=True)
    filepath_output = os.path.join(dirpath_output, f"{os.path.basename(filepath_metadata).split('.')[0]}_llavahf.json")
    context0 = {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "device": "cuda"
    }
    context = DotMap(context0)
    handler = LLaVAHFHandler(context)
    metadata = load_vindr_data(
        filepath_metadata=filepath_metadata
    )

    data_output = {}
    for filename_image, questions in tqdm(metadata.items()):
        try:
            data_output[filename_image] = []
            for sample in questions:
                filepath_image = os.path.join(dirpath_images, filename_image + ".png")
                question = sample["question"]
                question = question.replace('<image>', '').strip()
                handler_input = {
                    "filepath_image": filepath_image,
                    "user_prompt": question
                }
                handler_output = handler.handle(handler_input)
                sample["response"] = handler_output["response"]
                data_output[filename_image].append(sample)
                
                with open(filepath_output, "w") as f:
                    json.dump(data_output, f) 
        except Exception as e:
            print(f"Error: {e}. File: {filename_image}")
            continue


if __name__ == "__main__":
    evaluate_llavahf(
        filepath_metadata="/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/vqa_vindr_unhealthy_image_normal.json",
        dirpath_images="/mnt/12T/02_duong/data-center/VinDr/train_png_16bit",
        dirpath_output="/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/probmed_open_v2"
    )
