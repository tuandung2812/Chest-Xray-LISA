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
              'pneumothorax', 'pulmonary fibrosis']


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


def classify_diseases(data_predict):
    context = {
        # "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        # "model_id": "ruslanmv/Medical-Llama3-8B",
        "model_id": "m42-health/Llama3-Med42-8B",
        "device": "cuda"
    }
    evaluator = LLaMA3PostEvaluator(context)

    for idx, row in tqdm.tqdm(data_predict.iterrows(), total=len(data_predict)):
        messages_for_classify = [
            {
                "role": "system", 
                "content": f"""You are an expert doctor. 
                    You are not given image. The only information you have is <abnormality name>.
                    You need to classify the abnormality to only one of {", ".join(VINDR_LABEL)}.
                    Do not try to make your diagnosis based on the text
                    Return 'out of list' when the <abnormality name> is not in the list.
                    Remove anatomy information such as location.
                """
            },
            {
                "role": "user", 
                "content": "This is the <abnormality name> '{question}'"
            }
        ]
    
        try:
            answer = row['answer']
            prediction = row['response']
            prediction_extracted = row['prediction_extracted']


            if prediction_extracted in ["no finding", "uncertain", "ERROR"]:
                prediction_classified = prediction_extracted
            else:
                print("answer", answer)
                print("prediction", prediction)
                print("prediction_extracted", prediction_extracted)

                prediction_classified = []
                for disease in prediction_extracted.split(","):
                    if disease in VINDR_LABEL:
                        prediction_classified.append(disease)
                    else:
                        messages_new = messages_for_classify.copy()
                        messages_new[1]["content"] = messages_new[1]["content"].replace("{question}", disease)
                        disease_classified = evaluator.inference(
                            messages=messages_for_classify
                        )
                        print(disease, disease_classified)
                        prediction_classified.append(disease_classified)
                prediction_classified = ", ".join(prediction_classified)
            data_predict.loc[idx, 'prediction_classified'] = prediction_classified

            print("prediction_classified", prediction_classified)
            print()
        except Exception as e:
            print(e)
            print(f"Error at {idx}")
            data_predict.loc[idx, 'prediction_classified'] = "ERROR"
    
    return data_predict


def main(filename_predict, question_type):
    print(f"Post evaluate {filename_predict}")
    filepath_predict = os.path.join(DIRPATH_PREDICT, question_type, filename_predict)
    data_predict = pd.read_csv(filepath_predict)
    result = classify_diseases(data_predict)
    filepath_output = os.path.join(DIRPATH_PREDICT, question_type, filename_predict.replace("_post_extract.csv", "_post_classify.csv"))
    result.to_csv(filepath_output, index=False)


if __name__ == '__main__':
    # main(
    #     filename_predict="vindr_qa_data_test_1010_llavamed0.json",
    #     question_type="probmed/probmed_open"
    # )
    # main(
    #     filename_predict="vindr_qa_data_test_1010_llavahf.json",
    #     question_type="probmed/probmed_open"
    # )
    main(
        filename_predict="vindr_qa_data_test_1010_llavamed0_post_extract.csv",
        question_type="probmed/probmed_open",
    )
