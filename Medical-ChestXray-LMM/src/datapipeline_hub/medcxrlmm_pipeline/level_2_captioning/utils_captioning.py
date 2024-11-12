import sys
sys.path.append("/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/src")

import json
import re
from copy import deepcopy
import warnings
from pycocotools.coco import COCO
from pycocotools import mask

import numpy as np


def convert_mask_json2numpy(path):
    with open(path, 'r') as f:
        data0 = json.load(f)
    coco = COCO(path)
    ann_ids = coco.getAnnIds()
    masks = []
    for _, id in enumerate(ann_ids):
        annotation = coco.loadAnns([id])[0]
        mask = coco.annToMask(annotation)
        masks.append(mask)
    masks = np.stack(masks, axis=0)
    return data0, masks


def convert_box_txt2numpy(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.loadtxt(path, delimiter=' ')
    if len(data) == 0:
        return None
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    return data


def convert_caption_txt2dict(path):
    with open(path, 'r') as f:
        captions = f.readlines()[0]
    captions = captions.split(".")
    captions_list = [] 
    print(captions)
    for sentence in captions:
        print(sentence)
        caption_parsed = {}
        disease_match = re.search(r'<disease>(.*?)<box>(.*?)</box></disease>', sentence)
        anatomy_match = re.search(r'<anatomy>(.*?)</anatomy>', sentence)
        if not disease_match or not anatomy_match:
            continue
        caption_parsed["disease"] = disease_match
        caption_parsed["box"] = disease_match
        caption_parsed["anatomy"] = anatomy_match

        print(caption_parsed)
        captions_list.append(caption_parsed)


def rle_to_binary_mask(rle_mask):
    rle_mask_copy = deepcopy(rle_mask)
    rle_mask_copy['counts'] = rle_mask_copy['counts'].encode('utf-8')
    binary_mask = mask.decode(rle_mask_copy)
    return binary_mask
