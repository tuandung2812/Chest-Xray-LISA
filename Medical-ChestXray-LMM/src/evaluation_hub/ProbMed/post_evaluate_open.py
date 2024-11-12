import os
import json
import tqdm

import pandas as pd
import numpy as np

from huggingface_hub import login
import torch
import transformers

DIRPATH_PREDICT = "/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub" 

HF_TOKEN = os.environ.get("HF_TOKEN")
login(HF_TOKEN)
transformers.logging.set_verbosity_error()

VINDR_LABEL = ['aortic enlargement', 'atelectasis', 'calcification', 'cardiomegaly', 
              'consolidation', 'interstitial lung disease', 'infiltration', 'lung opacification', 'nodule/mass', 
              'other lesion', 'pleural effusion', 'pleural thickening', 
              'pneumothorax', 'pulmonary fibrosis', 'no finding']


class LLaMA3PostEvaluator():
    def __init__(self, context):
        self.initialize(context)

    def initialize(self, context):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context["model_id"]
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def inference(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1, 
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        prediction_extracted = self.tokenizer.decode(response, skip_special_tokens=True)
        return prediction_extracted


def load_filepath_predict(filepath_predict):
    with open(filepath_predict, "r") as f:
        data0 = json.load(f)
    records = []
    for key, values in data0.items():
        for value in values:
            record = {"id": key}
            record.update(value)
            records.append(record)
    data_predict = pd.DataFrame(records)
    return data_predict


def evaluate(data_predict):
    context = {
        # "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        # "model_id": "ruslanmv/Medical-Llama3-8B",
        "model_id": "m42-health/Llama3-Med42-8B",
        "device": "cuda"
    }
    evaluator = LLaMA3PostEvaluator(context)

    for idx, row in tqdm.tqdm(data_predict.iterrows(), total=len(data_predict)):
        messages_for_extract = [
            {
                "role": "system", 
                "content": f"""You are an expert chest xray radiologist. 
                    You are not given image. The only information you have is the text.
                    You need to extract only the name of abnormalities, a patient is suffer from in <medical report>.
                    Answer in this format: <abnormality 1>, <abnormality 2>, <abnormality 3>
                    Do not try to make your diagnosis based on the text
                    Answer "no finding" when patient is healthy or there is no specific name of abnormalities.
                    Remove anatomy information such as location.
                """
            },
            {
                "role": "user", 
                "content": "This is the <medical report> '{question}'"
            }
        ]
    
        try:
            answer = row['answer']
            prediction = row['response']

            if prediction in ["Yes", "yes"]:
                prediction_extracted = answer.replace("It suffers from ", "")
                assert prediction_extracted != "no abnormalities", prediction_extracted
                prediction_extracted_post = prediction_extracted.lower()
            else:
                messages_for_extract[1]["content"] = messages_for_extract[1]["content"].replace("{question}", prediction)
                prediction_extracted = evaluator.inference(
                    messages=messages_for_extract
                )
                prediction_extracted = prediction_extracted.strip().lower()
                prediction_extracted_new = []
                for disease in prediction_extracted.split(","):
                    if "in the" in disease:
                        disease = disease.split("in the")[0]
                    if " or " in disease:
                        disease = "uncertain"
                    prediction_extracted_new.append(disease.strip())
                prediction_extracted_post = ", ".join(prediction_extracted_new)
            data_predict.loc[idx, 'prediction_extracted'] = prediction_extracted_post
            # print(answer)
            # print(prediction)
            # print(prediction_extracted)
            # print(prediction_extracted_post)
        except Exception as e:
            print(e)
            print(f"Error at {idx}")
            data_predict.loc[idx, 'prediction_extracted'] = "ERROR"
    
    return data_predict


def post_evaluate_open(filename_predict, question_type):
    print(f"Post evaluate {filename_predict}")
    filepath_predict = os.path.join(DIRPATH_PREDICT, question_type, filename_predict)
    data_predict = load_filepath_predict(filepath_predict)
    result = evaluate(data_predict)
    filepath_post_eval = os.path.join(DIRPATH_PREDICT, question_type, filename_predict.replace(".json", "_post_eval.csv"))
    result.to_csv(filepath_post_eval, index=False)


if __name__ == '__main__':
    # post_evaluate_open(
    #     filename_predict="vindr_qa_data_test_1010_llavamed0.json",
    #     question_type="probmed/probmed_open"
    # )
    # post_evaluate_open(
    #     filename_predict="vindr_qa_data_test_1010_llavahf.json",
    #     question_type="probmed/probmed_open"
    # )
    post_evaluate_open(
        filename_predict="vindr_qa_data_test_1010_chexagenthf.json",
        question_type="probmed/probmed_open",
    )
