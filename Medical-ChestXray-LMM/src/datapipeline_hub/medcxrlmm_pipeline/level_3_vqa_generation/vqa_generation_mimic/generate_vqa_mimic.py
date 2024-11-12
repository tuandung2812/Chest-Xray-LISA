import sys
sys.path.append("/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/src")

import os
import json
import tqdm

import random

from datapipeline_hub.medcxrlmm_pipeline.level_3_vqa_generation.vqa_generation.extract_disease_anatomy_pairs import (
    diseases_radgraph, anatomies_radgraph, diseases_radgraph_ignored, anatomies_radgraph_ignored
)
from datapipeline_hub.medcxrlmm_pipeline.level_3_vqa_generation.vqa_generation.qa_format import *

diseases_radgraph_filtered = [disease for disease in diseases_radgraph if disease not in diseases_radgraph_ignored]
anatomies_radgraph_filtered = [anatomy for anatomy in anatomies_radgraph if anatomy not in anatomies_radgraph_ignored]
anatomies_target = [
    'left upper lung', 
    'left middle lung', 
    'left lower lung', 
    'right upper lung', 
    'right middle lung', 
    'right lower lung', 
    'mediastinum', 
    'aorta', 
    'spine', 
    'heart'
]


def main():
    dirpath_caption = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_MedGLaMM_caption_v3"
    filepath_mimic = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/src/data_pipeline_hub/COMG_Anatomy_pipeline/MIMIC_meta_data.json"
    with open(filepath_mimic) as file:
        data_mimic = json.load(file)
    data_mimic_split = data_mimic["train"] + data_mimic["valid"]
    data_mimic_dict = {}
    for sample in tqdm.tqdm(data_mimic_split):
        dicom_id = sample["dicom_id"]
        data_mimic_dict[dicom_id] = sample

    filepath_disease_anatomy_pairs = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/mimic_disease_anatomy_pairs.json"
    with open(filepath_disease_anatomy_pairs) as file:
        data_disease_anatomy_pairs = json.load(file)

    for dicom_id, data_mapping in tqdm.tqdm(data_disease_anatomy_pairs.items()):
        try:
            data_mimic = data_mimic_dict[dicom_id]
            filepath_caption = os.path.join(dirpath_caption, data_mimic["Path"].replace("2.0.0/files/", "").replace(".jpg", ".json"))
            data_out = data_mimic.copy()
            data_out["filepath_caption"] = filepath_caption
            data_out["question_answer_pair"] = []
            for pair in data_mapping:
                anatomy_name = pair["anatomy"]
                disease_name = pair["disease"]
                if anatomy_name in anatomies_target:
                    # yes no question
                    question_list_yesno = QUESTION_TEMPLATES["anatomy_disease_yes_no"]
                    question = random.choice(question_list_yesno).format(anatomy=anatomy_name)
                    answer = ANSWER_TEMPLATES["anatomy_disease_yes_no"].format(anatomy=anatomy_name)
                    data_out["question_answer_pair"].append({
                        "question_type": "anatomy_disease_yes_no",
                        "question": question,
                        "answer": answer
                    })
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    main()
