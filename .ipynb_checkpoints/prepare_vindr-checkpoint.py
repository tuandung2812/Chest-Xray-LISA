import json
import os
import numpy as np
from tqdm import tqdm
from pycocotools import mask
from skimage import measure

train_path = 'process_mimic/vindr_qa_data_train_newformat.json'
test_path = 'process_mimic/vindr_qa_data_test_newformat.json'
segment_path ='./dataset/VinDr/VinDr_MedGLaMM/train_png_16bit'


output_dir = './dataset/VinDr/annotation/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_output_dir = os.path.join(output_dir,'train')

if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)

test_output_dir = os.path.join(output_dir,'test')

if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

    
mask_dir = './dataset/VinDr/mask/'
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

train_mask_dir = os.path.join(mask_dir,'train')

if not os.path.exists(train_mask_dir):
    os.makedirs(train_mask_dir)

test_mask_dir = os.path.join(mask_dir,'test')

if not os.path.exists(test_mask_dir):
    os.makedirs(test_mask_dir)

ID_TO_ANATOMY_MAPPINGS = {
    0: 'Left upper lung',
    1: 'Left middle lung',
    2: 'Left lower lung',
    3: 'Right upper lung',
    4: 'Right middle lung',
    5: 'Right lower lung',
    6: 'Mediastinum',
    7: 'Aorta',
    8: 'Spine',
    9: 'Heart'
}

ANATOMY_TO_ID_MAPPINGS = {
    'Left upper lung': 0,
    'Left middle lung': 1,
    'Left lower lung': 2,
    'Right upper lung': 3,
    'Right middle lung': 4,
    'Right lower lung': 5,
    'Mediastinum': 6,
    'Aorta': 7,
    'Spine': 8,
    'Heart': 9
}
train_segment_data = []
with open(train_path) as f:
    train_data = json.load(f)
    
# print(train_data)
for qa in tqdm(train_data):
    image_id = qa['image_id']
    file_id = qa['id']
    
    file_dict = {}
    question, answer, anat = qa['question'], qa['grounded_answer'], qa['anat']
    
    file_dict['text'] = question
    file_dict['answer'] = answer
    file_dict['is_sentence'] = True
    file_dict['shapes'] = [ {
      "label": "target",
      "labels": [
        "target"
      ],
      "shape_type": "polygon",
      "image_name": f"{image_id}.jpg"}]
    anat_id = ANATOMY_TO_ID_MAPPINGS[anat]
    if '-checkpoint' in image_id:
        image_id = image_id.replace('-checkpoint','')
    segment_file = os.path.join(segment_path, f'{image_id}.json')
    # print(segment_file)
    with open(segment_file) as f:
        segment_data = json.load(f)
    for annotation in segment_data['annotations']:
        # print(annotation)
        if annotation['category_id'] == anat_id:
            rle_mask = annotation['segmentation']
            binary_mask = mask.decode(rle_mask)
            mask_path = os.path.join(train_mask_dir, f'{file_id}.npy')
            np.save(mask_path, binary_mask)

            try:
                contours = measure.find_contours(binary_mask, level=0.5)[0]
                points = contours.tolist()
                file_dict['shapes'][0]['points'] = points
                with open(os.path.join(train_output_dir,f'{file_id}.json'),'w') as f:
                    json.dump(file_dict,f)
            except:
                pass
                    # print(contours)
            # print(binary_mask.shape)
            # print(anat)
            # print(annotation)
    # print(qa)
    # print(qa)
    # break
    
    # print(segment_data)
    # print(seg
    # break
test_segment_data = []
with open(test_path) as f:
    test_data = json.load(f)

for qa in tqdm(test_data):
    image_id = qa['image_id']
    file_id = qa['id']
    
    file_dict = {}
    question, answer, anat = qa['question'], qa['grounded_answer'], qa['anat']
    
    file_dict['text'] = question
    file_dict['answer'] = answer
    file_dict['is_sentence'] = True
    file_dict['shapes'] = [ {
      "label": "target",
      "labels": [
        "target"
      ],
      "shape_type": "polygon",
      "image_name": f"{image_id}.jpg"}]
    anat_id = ANATOMY_TO_ID_MAPPINGS[anat]
    if '-checkpoint' in image_id:
        image_id = image_id.replace('-checkpoint','')
    segment_file = os.path.join(segment_path, f'{image_id}.json')
    # print(segment_file)
    with open(segment_file) as f:
        segment_data = json.load(f)
    for annotation in segment_data['annotations']:
        # print(annotation)
        if annotation['category_id'] == anat_id:
            rle_mask = annotation['segmentation']
            binary_mask = mask.decode(rle_mask)
            mask_path = os.path.join(test_mask_dir, f'{file_id}.npy')
            np.save(mask_path, binary_mask)
            try:
                contours = measure.find_contours(binary_mask, level=0.5)[0]
                # print(contours)
                # points = contours
                points = contours.tolist()
                file_dict['shapes'][0]['points'] = points
                with open(os.path.join(test_output_dir,f'{file_id}.json'),'w') as f:
                    json.dump(file_dict,f)
            except:
                pass