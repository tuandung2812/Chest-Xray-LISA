import sys
sys.path.append(".")

import os
from dotmap import DotMap
from tqdm import tqdm
import json

from evaluation_hub.load_data.load_probmed_data import load_probmed_data


def evaluate_llava0(filepath_metadata,
    dirpath_images,
    dirpath_output,
):  
    os.makedirs(dirpath_output, exist_ok=True)
    filepath_output = os.path.join(dirpath_output, f"{os.path.basename(filepath_metadata).split('.')[0]}_llava0.json")
    context0 = {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "device": "cuda"
    }
    context = DotMap(context0)
    metadata = load_probmed_data(
        filepath_metadata=filepath_metadata
    )

    data_output = []
    for sample0 in tqdm(metadata):
        pass


if __name__ == '__main__':
    evaluate_llava0(
        filepath_metadata="/mnt/12T/02_duong/data-center/ProbMed/test/test.json",
        dirpath_images="/mnt/12T/02_duong/data-center/ProbMed/test",
        dirpath_output="/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/vlm_visualizer"
    )