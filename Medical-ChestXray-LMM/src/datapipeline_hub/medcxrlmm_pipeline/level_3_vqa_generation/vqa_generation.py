import json
import tqdm

import random

from question_answer_formats.qa_format_radgenome import *
from question_answer_formats.qa_format_basic import *

VINDR_LABEL = ['aortic enlargement', 'atelectasis', 'calcification', 'cardiomegaly', 
              'consolidation', 'interstitial lung disease', 'infiltration', 'lung opacification', 'nodule/mass', 
              'other lesion', 'pleural effusion', 'pleural thickening', 
              'pneumothorax', 'pulmonary fibrosis']

REGION_LABEL = {
    "left upper lung": 0,  # CheXmask
    "left middle lung": 1,  # CXAS
    "left lower lung": 2,  # CheXmask
    "right upper lung": 3,  # CheXmask
    "right middle lung": 4,  # CXAS
    "right lower lung": 5,  # CheXmask
    "mediastinum": 6,  # CXAS
    "aorta": 7,  # CXAS
    "spine": 8,  # ChestX
    "heart": 9,  # CheXmask
}


def gen_vqa_unhealthy_normal():
    filepath_vqa = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/vindr_qa_data_test_1010.json"
    with open(filepath_vqa, "r") as f:
        metadata = json.load(f)

    vqa_output = {}
    for filename_image, questions in tqdm.tqdm(metadata.items()):
        try:
            for sample in questions:
                if sample["type"] != "anatomy_open":
                    continue
                vqa_output[filename_image] = []
                abnormality = sample["answer"].replace("It suffers from ", "").lower()
                assert abnormality != "no abnormalities"
                if abnormality == "ild":
                    abnormality = "interstitial lung disease"
                region = sample["anat"].lower()

                abnormality_question = random.choice(list(BASIC_ABNORMALITY_QUESTION_DICT.values())).format(region=region)
                vqa_output[filename_image].append({
                    "question_type": "abnormality",
                    "question": f"Look at the {region}. " + abnormality_question,
                    "region": region,
                    "answer": f"The location of {abnormality} is at <seg>.",
                })

                presense_question = random.choice(list(BASIC_PRESENCE_QUESTION_DICT.values())).format(abnormality=abnormality, region=region)
                vqa_output[filename_image].append({
                    "question_type": "presence",
                    "question": f"Look at the {region}. " + presense_question,
                    "answer": f"The location of {abnormality} is at <seg>. The answer is Yes",
                })
                
                location_question = random.choice(list(BASIC_LOCATION_QUESTION_DICT.values())).format(
                    abnormality=abnormality,
                    region=region
                )
                vqa_output[filename_image].append({
                    "question_type": "location",
                    "question": f"Look at the {region}. " + location_question,
                    "answer": region
                })

                abnormality_question_open = random.choice(list(BASIC_ABNORMALITY_QUESTION_DICT.values())).format(region="image")
                vqa_output[filename_image].append({
                    "question_type": "abnormality_open",
                    "question": f"Look at the {region}. " + abnormality_question_open,
                    "region": "image",
                    "answer": abnormality,
                })

                # presense_question_open = random.choice(list(BASIC_PRESENCE_QUESTION_DICT.values())).format(abnormality="an abnormality", region=region)
                # vqa_output[filename_image].append({
                #     "question_type": "presence_open_having_region",
                #     "question": f"Look at the {region}. " + presense_question_open,
                #     "answer": "yes"
                # })

                # presense_question_open = random.choice(list(BASIC_PRESENCE_QUESTION_DICT.values())).format(abnormality="an abnormality", region="image")
                # vqa_output[filename_image].append({
                #     "question_type": "presence_open_having_no_region",
                #     "question": f"Look at the {region}. " + presense_question_open,
                #     "answer": "yes"
                # })

                location_question_open = random.choice(list(BASIC_LOCATION_QUESTION_DICT.values())).format(
                    abnormality="an abnormality",
                    region=region
                )
                vqa_output[filename_image].append({
                    "question_type": "location_open",
                    "question": f"Look at the {region}. " + location_question_open,
                    "answer": region
                })
        except Exception as e:
            print(f"Error: {e}. File: {filename_image}")
            continue

    filepath_vqa_output = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/vqa_vindr_unhealthy_image_normal_new.json"
    with open(filepath_vqa_output, "w") as f:
        json.dump(vqa_output, f, indent=4)


def gen_vqa_unhealthy_adversarial():
    filepath_vqa = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/vindr_qa_data_test_1010.json"
    with open(filepath_vqa, "r") as f:
        metadata = json.load(f)

    vqa_output = {}
    for filename_image, questions in tqdm.tqdm(metadata.items()):
        try:
            for sample in questions:
                if sample["type"] != "anatomy_open":
                    continue
                vqa_output[filename_image] = []
                abnormality = sample["answer"].replace("It suffers from ", "").lower()
                assert abnormality != "no abnormalities"
                region = sample["anat"].lower()

                # make up region
                makeup_region = random.choice([label for label in REGION_LABEL if label != region])
                abnormality_question = random.choice(list(BASIC_ABNORMALITY_QUESTION_DICT.values())).format(region=makeup_region)
                vqa_output[filename_image].append({
                    "question_type": "abnormality",
                    "question": f"Look at the {makeup_region}. " + abnormality_question,
                    "region": makeup_region,
                    "answer": "no finding",
                })

                # make up abnormality and region
                makeup_abnormality = random.choice([label for label in VINDR_LABEL if label != abnormality])
                makeup_region = random.choice([label for label in REGION_LABEL if label != region])
                presense_question = random.choice(list(BASIC_PRESENCE_QUESTION_DICT.values())).format(abnormality=makeup_abnormality, region=makeup_region)
                vqa_output[filename_image].append({
                    "question_type": "presence",
                    "question": f"Look at the {makeup_region}. " + presense_question,
                    "answer": "no"
                })

                # make up abnormality
                makeup_abnormality = random.choice([label for label in VINDR_LABEL if label != abnormality])
                location_question = random.choice(list(BASIC_LOCATION_QUESTION_DICT.values())).format(
                    abnormality=makeup_abnormality,
                    region=region,
                )
                vqa_output[filename_image].append({
                    "question_type": "location",
                    "question": f"Look at the {makeup_region}. " + location_question,
                    "answer": "no finding",
                })
        except Exception as e:
            print(f"Error: {e}. File: {filename_image}")
            continue

    filepath_vqa_output = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/vqa_vindr_unhealthy_image_makeup_new.json"
    with open(filepath_vqa_output, "w") as f:
        json.dump(vqa_output, f, indent=4)


if __name__ == "__main__":
    gen_vqa_unhealthy_normal()
    gen_vqa_unhealthy_adversarial()
