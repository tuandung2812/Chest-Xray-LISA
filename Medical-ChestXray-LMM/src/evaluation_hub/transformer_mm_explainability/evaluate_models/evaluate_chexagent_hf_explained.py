import sys
sys.path.append(".")

import os
from dotmap import DotMap
from tqdm import tqdm

import transformers

from model_hub.chexagent.chexagent_hf_explained.predict_chexagent_explained import ChexagentExplained_Handler
from evaluation_hub.load_data.load_probmed_data import load_probmed_data

transformers.logging.set_verbosity_error()


def evaluate_chexagent(
    filepath_metadata,
    dirpath_images,
    dirpath_output,
):  
    os.makedirs(dirpath_output, exist_ok=True)
    context0 = {
        "device": "cuda",
        "model_id": "StanfordAIMI/CheXagent-8b"
    }
    context = DotMap(context0)
    handler = ChexagentExplained_Handler(context)
    metadata = load_probmed_data(
        filepath_metadata=filepath_metadata
    )

    for sample0 in tqdm(metadata):
        sample = sample0
        if sample["image_type"] != "x-ray_chest":
            continue
        filepath_image = os.path.join(dirpath_images, sample["image"])
        question = sample["question"].replace('<image>', '').strip()
        handler_input = {
            "filepath_image": filepath_image,
            "user_prompt": question
        }
        handler_output = handler.handle(handler_input)
    

if __name__ == "__main__":
    evaluate_chexagent(
        filepath_metadata="/mnt/12T/02_duong/data-center/ProbMed/test/test.json",
        dirpath_images="/mnt/12T/02_duong/data-center/ProbMed/test",
        dirpath_output="/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/transformer_mm_explainability"
    )
