import os
import json
import tqdm
import re

import pandas as pd

from huggingface_hub import login
import torch
import transformers

from evaluation_hub.probmed.utils.calculate_metrics import calculate_yesno

DIRPATH_PREDICT = "/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub" 

HF_TOKEN = os.environ.get("HF_TOKEN")
login(HF_TOKEN)
transformers.logging.set_verbosity_error()


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

    def inference(self, question):
        self.messages = [
            {
                "role": "system", 
                "content": f"""You are an expert Radiologist
                    You are not given image. The only information you have is the text
                    You need to determine if the patient is suffer from abnormalities or not based on the text
                    Answer in this format: 'Yes' or 'No'
                    Answer "Uncertain" when you are uncertain
                """
            },
            {
                "role": "user", 
                "content": f"This is the medical report '{question}'"
            }
        ]

        input_ids = self.tokenizer.apply_chat_template(
            self.messages,
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
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "device": "cuda"
    }
    evaluator = LLaMA3PostEvaluator(context)

    uncertain_cases = ["Yes or No", "Yes/No", "Yes/No: Yes", "Yes/No: No", "Yes / No", "'yes' or 'no'"]

    for idx, row in tqdm.tqdm(data_predict.iterrows(), total=len(data_predict)):
        try:
            prediction_extracted = None
            prediction = row['response']
            prediction = re.sub(r'[^a-zA-Z\s,\./]', '', prediction)
            prediction = prediction.strip()
            for case in uncertain_cases:
                if case.lower() in prediction.lower():
                    prediction_extracted = "Uncertain"
                    break
            if prediction_extracted == "Uncertain":
                pass
            elif prediction in ["Yes", "No", "Yes.", "No."]:
                prediction_extracted = prediction.replace(".", "")
            else:
                prediction_extracted = evaluator.inference(prediction)
            assert prediction_extracted in ["Yes", "No", "Uncertain"], f"Prediction extracted: {prediction_extracted}, Prediction: {prediction}"
            data_predict.loc[idx, 'prediction_extracted'] = prediction_extracted
        except Exception as e:
            print(e)
            print(f"Error at {idx}")
            data_predict.loc[idx, 'prediction_extracted'] = "ERROR"
    return data_predict


def post_evaluate_yesno(filename_predict, question_type):
    filepath_predict = os.path.join(DIRPATH_PREDICT, question_type, filename_predict)
    data_predict = load_filepath_predict(filepath_predict)
    result = evaluate(data_predict)
    filepath_post_eval = os.path.join(DIRPATH_PREDICT, question_type, filename_predict.replace(".json", "_post_eval.csv"))
    result.to_csv(filepath_post_eval, index=False)

    calculate_yesno(
        filepath_predict=filepath_post_eval,
        groupby="anatomy_disease_yes_no"
    )
    calculate_yesno(
        filepath_predict=filepath_post_eval,
        groupby="adversarial_anatomy_disease_yes_no"
    )
    calculate_yesno(
        filepath_predict=filepath_post_eval
    )


if __name__ == '__main__':
    post_evaluate_yesno(
        filename_predict="vindr_qa_data_test_lisa.json",
        question_type="probmed/probmed_yesno"
    )

    # post_evaluate_yesno(
    #     filename_predict="vindr_qa_data_test_llavahf.json",
    #     question_type="probmed/probmed_yesno"
    # )

    # post_evaluate_yesno(
    #     filename_predict="vindr_qa_data_test_chexagenthf.json",
    #     question_type="probmed/probmed_yesno"
    # )
