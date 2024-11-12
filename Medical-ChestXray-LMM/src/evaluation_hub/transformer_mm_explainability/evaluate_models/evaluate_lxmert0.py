import sys
sys.path.append(".")

import os
from dotmap import DotMap
from tqdm import tqdm

from evaluation_hub.load_data.load_probmed_data import load_probmed_data
from model_hub.lxmert.lxmert_0_explained.predict_lxmert_0_explained import LXMERTExplainedHandler


def evaluate_lxmert0(
    filepath_metadata,
    dirpath_images,
    dirpath_output,
):
    os.makedirs(dirpath_output, exist_ok=True)
    filepath_output = os.path.join(dirpath_output, f"{os.path.basename(filepath_metadata).split('.')[0]}_lxmert0.json")
    context0 = {
        "model_id": "lxmert_0",
        "use_lrp": True,
    }
    context = DotMap(context0)
    handler = LXMERTExplainedHandler(context)
    metadata = load_probmed_data(
        filepath_metadata=filepath_metadata
    )

    data_output = []
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
        R_t_t, R_t_i = handler.handle(handler_input)
        visualize(R_t_t, R_t_i, handler.model, dirpath_output, filepath_image, question)
        

def visualize(
    R_t_t, 
    R_t_i,
    model_lrp,
    dirpath_output,
    filepath_image,
    question
):
    from matplotlib import pyplot as plt
    from PIL import Image
    import numpy as np
    import cv2
    import torch
    from captum.attr import visualization
    from model_hub.lxmert.lxmert_0_explained.modeling import vqa_utils as utils

    image_scores = R_t_i[0]
    text_scores = R_t_t[0]
    VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
    vqa_answers = utils.get_data(VQA_URL)

    def save_image_vis(image_file_path, bbox_scores):
        bbox_scores = image_scores
        _, top_bboxes_indices = bbox_scores.topk(k=1, dim=-1)
        img = cv2.imread(image_file_path)
        img0 = img.copy()
        mask = torch.zeros(img.shape[0], img.shape[1])
        for index in range(len(bbox_scores)):
            [x, y, w, h] = model_lrp.bboxes[0][index]
            curr_score_tensor = mask[int(y):int(h), int(x):int(w)]
            new_score_tensor = torch.ones_like(curr_score_tensor)*bbox_scores[index].item()
            mask[int(y):int(h), int(x):int(w)] = torch.max(new_score_tensor,mask[int(y):int(h), int(x):int(w)])
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = mask.unsqueeze_(-1)
        mask = mask.expand(img.shape)
        img = img * mask.cpu().data.numpy()
        concat = np.concatenate((img0, img), axis=1)
        cv2.imwrite(os.path.join(dirpath_output, 'new.jpg'), concat)
    
    save_image_vis(filepath_image, image_scores)
    orig_image = Image.open(model_lrp.image_file_path)

    fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
    axs[0].imshow(orig_image)
    axs[0].axis('off')
    axs[0].set_title('original')
    [[']]]]]']]
    masked_image = Image.open(os.path.join(dirpath_output, 'new.jpg'))
    axs[1].imshow(masked_image)
    axs[1].axis('off')
    axs[1].set_title('masked')

    text_scores = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min())
    vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,model_lrp.question_tokens,1)]
    visualization.visualize_text(vis_data_records)
    print(question)
    print("ANSWER:", vqa_answers[model_lrp.output.question_answering_score.argmax()])


if __name__ == '__main__':
    evaluate_lxmert0(
        filepath_metadata="/mnt/12T/02_duong/data-center/ProbMed/test/test.json",
        dirpath_images="/mnt/12T/02_duong/data-center/ProbMed/test",
        dirpath_output="/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/transformer_mm_explainability"
    )
