import tqdm
import json

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# VINDR_LABEL = ['aortic enlargement', 'atelectasis', 'calcification', 'cardiomegaly', 
#               'consolidation', 'interstitial lung disease', 'infiltration', 'lung opacification', 'nodule/mass', 
#               'other lesion', 'pleural effusion', 'pleural thickening', 
#               'pneumothorax', 'pulmonary fibrosis']


VINDR_LABEL = {
    "no finding": 0,
    "aortic enlargement": 1,
    "atelectasis": 2,
    "calcification": 3,
    "cardiomegaly": 4,
    "consolidation": 5,
    "interstitial lung disease": 6,
    "infiltration": 7,
    "lung opacity": 8,
    "nodule/mass": 9,
    "other lesion": 10,
    "pleural effusion": 11,
    "pleural thickening": 12,
    "pneumothorax": 13,
    "pulmonary fibrosis": 14,
}


def main():
    filepath_prediction_extracted = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/src/evaluation_hub/probmed/post_evaluation/llavamed0_gpt_abnormality.tsv"
    df_extracted = pd.read_csv(filepath_prediction_extracted, sep="\t")

    with open("/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/probmed_v2/vqa_vindr_unhealthy_image_normal_llavamed0.json", "r") as f:
        df_predicted = json.load(f)

    y_true = []
    y_pred = []

    for i, row in tqdm.tqdm(df_extracted.iterrows(), total=len(df_extracted)):
        try:
            data_image = df_predicted[row["id"]]
            for sample in data_image:
                if sample["response"] == row["pred"]:
                    label = sample["answer"]
                    y_true_sample = []
                    for k, v in VINDR_LABEL.items():
                        if k == label:
                            y_true_sample.append(1)
                        else:
                            y_true_sample.append(0)
                    
                    y_pred_sample = [0] * len(VINDR_LABEL)
                    diseases_predicted = row["result"]
                    diseases_predicted = diseases_predicted.split(",")
                    for disease_pred in diseases_predicted:
                        y_pred_sample[VINDR_LABEL[disease_pred]] = 1

                    y_true.append(y_true_sample)
                    y_pred.append(y_pred_sample)
                    pass

        except Exception as e:
            # print(e)
            # print(row)
            continue


    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    f1_manual_list = []
    for i in range(y_true_np.shape[1]):  # Iterate over columns
        true_labels = y_true_np[:, i]
        pred_labels = y_pred_np[:, i]
        
        # Calculate metrics for each class
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0, average="binary")

        f1_manual = 2 * precision * recall / (precision + recall)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        f1_manual_list.append(f1_manual)

    # Display metrics for each class
    for i in range(y_true_np.shape[1]):
        print(f'Class {i}: {list(VINDR_LABEL.keys())[i]}')
        print(f'  Accuracy: {accuracies[i]}')
        print(f'  Precision: {precisions[i]}')
        print(f'  Recall: {recalls[i]}')
        print(f'  F1 Score: {f1_scores[i]}')
        print(f'  F1 Manual: {f1_manual_list[i]}')
        print()


    
if __name__ == "__main__":
    main()
