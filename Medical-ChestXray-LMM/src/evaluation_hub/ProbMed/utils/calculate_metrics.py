import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calculate_yesno(filepath_predict, groupby=None):
    data_predict = pd.read_csv(filepath_predict)
    if groupby:
        data_predict = data_predict[data_predict['type'] == groupby]
    data_predict = data_predict[data_predict['prediction_extracted'] != "Uncertain"]
    data_predict = data_predict[data_predict['prediction_extracted'] != "ERROR"]

    if groupby == "adversarial_anatomy_disease_yes_no":
        # data_predict = data_predict.sample(frac=1).reset_index(drop=True)
        data_predict = data_predict[:1659]
    elif groupby is None:
        data_predict_pos = data_predict[data_predict['type'] == "anatomy_disease_yes_no"]
        data_predict_adversarial = data_predict[data_predict['type'] == "adversarial_anatomy_disease_yes_no"]
        # data_predict_adversarial = data_predict_adversarial.sample(frac=1).reset_index(drop=True)
        data_predict_adversarial = data_predict_adversarial[:1659]
        data_predict = pd.concat([data_predict_pos, data_predict_adversarial])

    data_predict['correct'] = data_predict['answer'] == data_predict['prediction_extracted']
    accuracy = data_predict['correct'].mean()
    print(f"Accuracy: {accuracy}")

    data_predict['answer'] = data_predict['answer'].apply(lambda x: 1 if x == "Yes" else 0)
    data_predict['prediction_extracted'] = data_predict['prediction_extracted'].apply(lambda x: 1 if x == "Yes" else 0)
    tp = data_predict[(data_predict['answer'] == 1) & (data_predict['prediction_extracted'] == 1)].shape[0]
    fp = data_predict[(data_predict['answer'] == 0) & (data_predict['prediction_extracted'] == 1)].shape[0]
    fn = data_predict[(data_predict['answer'] == 1) & (data_predict['prediction_extracted'] == 0)].shape[0]
    tn = data_predict[(data_predict['answer'] == 0) & (data_predict['prediction_extracted'] == 0)].shape[0]
    precision = tp / (tp + fp)
    if data_predict['answer'].sum() == 0:
        print("No positive case -> Recall and F1 are ignored")
        recall = "Ignored"
        f1_score = "Ignored"
    else:
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
    print(f"Number of positive cases: {data_predict['answer'].sum()}")
    print(f"Number of negative cases: {data_predict.shape[0] - data_predict['answer'].sum()}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print()


def calculate_open(y_true, y_pred):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Calculate metrics for each class (column)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    for i in range(y_true_np.shape[1]):  # Iterate over columns
        true_labels = y_true_np[:, i]
        pred_labels = y_pred_np[:, i]
        
        # Calculate metrics for each class
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Display metrics for each class
    for i in range(y_true_np.shape[1]):
        print(f'Class {i}:')
        print(f'  Accuracy: {accuracies[i]}')
        print(f'  Precision: {precisions[i]}')
        print(f'  Recall: {recalls[i]}')
        print(f'  F1 Score: {f1_scores[i]}')
        print()
