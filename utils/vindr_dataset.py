import glob
import json
import os
import random
import sys
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import transformers
# Add the LisaMed directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from model.llava import conversation as conversation_lib
# from model.segment_anything.utils.transforms import ResizeLongestSide

# from .data_processing import get_mask_from_json
# from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
#                     EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
#                     SHORT_QUESTION_LIST)


from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

# from .data_processing import get_mask_from_json
# from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
#                     EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
#                     SHORT_QUESTION_LIST)

# from utils.data_processing import get_mask_from_json
# from utils.utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
#                     EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
#                     SHORT_QUESTION_LIST)


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence

class VinDrDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    
    def __init__(
        self,
        base_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        split="train",
        text_only = False
    ):
        self.exclude_val = exclude_val
        self.base_dir = base_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        # self.explanatory = explanatory
        # self.num_classes_per_sample = num_classes_per_sample
        # vindr_seg_data=vindr_seg_data.split("|")
        splits = split.split("_")
        self.split = split
        self.text_only=  text_only

        jsons = []
        for split in splits:
            # print('vin_dr_splits: ', split)
            jsons_split = glob.glob(
                os.path.join(
                    base_dir, 'annotation_final', split, "*.json"
                )
            )
            jsons.extend(jsons_split)
        # print(jsons)
        self.vindr_data = []
        for json_path in jsons:
            json_id = json_path.split('/')[-1].replace('.json','')
            image_id = json_id.split('_')[-1]
            if '-checkpoint' in image_id:
                image_id = image_id.replace('-checkpoint','')

            image_path = os.path.join(self.base_dir, "image/jpg/all", image_id + '.jpg')
            
            mask_path = os.path.join(self.base_dir, "mask_final", split, json_id + '.npy')
            # if not os.path.exists(mask_path):
            #     # print(mask_path)
            self.vindr_data.append({'annotation_path':json_path, 'image_path':image_path, 'mask_path': mask_path})
        # print(self.vindr_data)
            
        # print(len(self.vindr_data))
        # jsons = []
        # for image_path in images:
        #     # print(image_path)
        #     print(image_path.replace('image/jpg', 'annotation').replace('jpg','json'))
        # print(images)
        
    def __len__(self):
        return len(self.vindr_data)
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # print('current idx: ', idx)
        sample = self.vindr_data[idx]
        img_path, json_path , mask_path = sample['image_path'], sample['annotation_path'], sample['mask_path']
        # print(img_path, mask_path)
        # print(img_path, json_path, mask_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ori_image = image
        image = np.stack([image] * 3, axis=-1)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"][0]
        # print('image', image.shape)
        # print('image_clip', image_clip.shape)
        # print('json_path: ', json_path)
        with open(json_path) as f:
            annotation = json.load(f)
        # print(annotation)
        question, answer = annotation['text'], annotation['answer']        
        answer = answer.replace('<seg>','[SEG]')
        questions = [DEFAULT_IMAGE_TOKEN + "\n" + question + '  Please first respond with segmentation mask, then provide the answer to the question.']
        answers = [answer]
        # print(question, answer)
        
        mask = np.load(mask_path)

        image = self.transform.apply_image(image)  # preprocess image for sam
        # print('2nd image', image.shape)
        resize = image.shape[:2]
        
        conversations = []
        conv = conversation_lib.default_conversation_medical.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], questions[0])
        conv.append_message(conv.roles[1], answers[0])
        conversations.append(conv.get_prompt())
        # print('Proconv.get_prompt())
        # print(conversations)

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # print(image.shape)
        masks = np.stack([mask], axis= 0)

        # print(masks.shape)
        # print(masks)
        # masks = np.stack([mask] * 3, axis=0)
        masks = torch.from_numpy(masks)

        # print(masks.shape)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        conversations = conversations 
        if self.split == 'train':
            inference = False
        else:
            inference = True
        return (
            img_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            None,
            inference,
            self.text_only
        )

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    output_dir = 'viz'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER = "<image-placeholder>"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
    "microsoft/llava-med-v1.5-mistral-7b",
        cache_dir=None,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,)
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")

    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )

    print(tokenizer.decode(torch.Tensor([415,  4723,
           302, 14552,   349,   438, 28705, 32000,   842,   415,  4372,   349,
          1770,     2])))
    # import pdb; pdb.set_trace()
    print(tokenizer.decode(torch.Tensor([1,   995,   460,   264, 10865,  8118,  1318, 28733,   919,  2996,
         24402, 13892, 28723,   259,    13,  2287,   415, 13892, 10148, 12189,
           272,  2078,  3036,   302,   272,  3469, 28723,   415, 13892,   285,
          2262,  9051,   272,  4424,   302,  2145,   304, 20105,   272,  2698,
           302,  2145, 28723,  2530,  2686, 28725,   378,  5212, 10865, 28725,
         11229, 28725,  3078,   864, 11194,   298,   272,  2188, 28742, 28713,
          4224, 28723,   260, 16693, 28705, 28740, 28747,    13,  2287,   387,
         28705, 32001, 32002, 28705,    13,  5985,   736,   707, 13875,
           438, 18365,  4986, 14966, 28804, 28705,    13,  2287,   387,   415,
          4723,   302, 18365,  4986, 14966,   349,   438, 28705, 32000,   842,
           661,  3510,   404,   477, 23436,  1890,  6931,  3250,    13,  2287,
         16693, 28705, 28750, 28747,    13,  2287,   387, 28705, 32001, 
         32002, 28705,    13, 20510,   330,   419, 28708, 13572,   477,  2984,
         28717,  2500, 28804, 28705,    13,  2287,   387,   415,  4723,   302,
           330,   419, 28708,   349,   438, 28705, 32000,   842,   415,  4372,
           349,  1770,    13,  2223,   725, 28747, 28705, 32001, 32002,
         28705,    13, 20510,   272,  7749, 28742, 28713,   330,   419, 28708,
           506,   707,   534,  8027,  1218, 28804,  8602,  8048, 12738, 28747,
           415,  4723,   302,   330,   419, 28708,   349,   438, 28705, 32000,
           842,   661,  3510,   404,   477,   330,   419,   294,   481, 20832,
          1116,     2]), special_tokens=True))
    dataset = VinDrDataset(base_dir='./dataset/VinDr',tokenizer=tokenizer, vision_tower="openai/clip-vit-large-patch14", split = 'train')
    alpha = 0.5
    i = 0
    for data in tqdm(dataset):
        # print(data)
        # if i == 70:
        #     break
        
        # print(data)
        # print('conversation: ', data[3])
        img_path = data[0]
        image = data[1][0]
        img_id = img_path.split('/')[-1]
        mask = data[4][0]
        print(mask.shape, image.shape)
        question = data[7][0]
        # print(img_path, img_id, mask)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # binary_mask = np.load(mask_numpy_path)
        print('image: ', image.shape)
        image = np.stack([image] * 3, axis=-1)
        mask_colored = np.zeros_like(image)
        # print(mask.shape, image.shape)
        mask_colored[mask == 1] = [255, 0, 0]  # Gán màu đỏ cho vùng mặt nạ (RGB: [255, 0, 0])
#         overlay_image = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
#         # Chèn văn bản vào ảnh
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         position = (50, 50)  # Tọa độ để chèn văn bản
#         font_scale = 1.0
#         font_color = (255, 255, 255)  # Màu trắng
#         thickness = 2
#         line_type = cv2.LINE_AA

#         # Thêm văn bản vào ảnh
#         overlay_image_with_text = cv2.putText(overlay_image, question, position, font, font_scale, font_color, thickness, line_type)
        
#         output_path = os.path.join(output_dir, img_id)
#         # cv2.imwrite(output_path, overlay_image_with_text)

#         # Hiển thị ảnh với overlay và văn bản
#         plt.figure(figsize=(8, 8))
#         plt.imshow(overlay_image_with_text)
#         plt.title("Grayscale Image with Segmentation and Text")
#         plt.axis('off')
#         plt.show()
        
#         i += 1