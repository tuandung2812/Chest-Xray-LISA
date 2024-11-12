import json
import os
import numpy as np
from tqdm import tqdm
from pycocotools import mask
from skimage import measure

train_path = 'process_mimic/mimic_qa_train_resampled.json'
test_path = 'process_mimic/mimic_qa_test_resampled.json'
segment_path ='/home/user01/aiotlab/dung_paper/groundingLMM/LISAMed/dataset/mimic'


output_dir = './dataset/mimic_data/annotation_resampled_new_3/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_output_dir = os.path.join(output_dir,'train')

if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)

test_output_dir = os.path.join(output_dir,'test')

if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

    
mask_dir = './dataset/mimic_data/mask_resampled_new_3/'
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


MIMIC_MAPPINGS = {
    "pleural unspec": ["Left upper lung", "Right upper lung", "Left middle lung", "Right middle lung", "Left lower lung", "Right lower lung"],
    "left pleural": ["Left upper lung", "Left middle lung", "Left lower lung"],
    "left lung unspec": ["Left upper lung", "Left middle lung", "Left lower lung"],
    "right lung unspec": ["Right upper lung", "Right middle lung", "Right lower lung"],
    "left diaphragm": ["Left lower lung"],
    "pulmonary": ["Right upper lung", "Left upper lung", "Right middle lung", "Left middle lung", "Right lower lung", "Left lower lung", "Heart"],
    "hilar unspec": ["Right upper lung", "Left upper lung", "Right middle lung", "Left middle lung", "Heart"],
    "lung volumes": ["Right upper lung", "Left upper lung", "Right middle lung", "Left middle lung", "Right lower lung", "Left lower lung"],
    "right pleural": ["Right upper lung", "Right middle lung", "Right lower lung"],
    "right diaphragm": ["Right lower lung"],
    "diaphragm unspec": ["Left lower lung", "Right lower lung", "Spine"],
    "lung bases": ["Right lower lung", "Left lower lung"],
    "right hilar": ["Right upper lung", "Right middle lung", "Heart"],
    "left hilar": ["Left upper lung", "Left middle lung", "Heart"],
    "retrocardiac": ["Heart"],
    "svc": ["Heart", "Right upper lung"],
    "cavoatrial junction": ["Heart", "Right upper lung"],
    "cardiophrenic sulcus": ["Heart"],
    "costophrenic unspec": ["Left lower lung", "Right lower lung"],
    "left costophrenic": ["Left lower lung"],
    "right costophrenic": ["Right lower lung"]
}
def merge_rle_encodings(rle_list):
    # Decode each RLE mask into a binary mask
    masks = [mask.decode(rle).astype(bool) for rle in rle_list]
    
    # Combine the binary masks into a single binary mask (logical OR)
    merged_mask = np.any(masks, axis=0)
    
    # Encode the merged mask back to RLE
    # merged_rle = mask_utils.encode(np.asfortranarray(merged_mask.astype(np.uint8)))
    
    return merged_mask

def merge_rle_encodings(rle_list):
    # Decode each RLE mask into a binary mask
    masks = [mask.decode(rle).astype(bool) for rle in rle_list]
    
    # Combine the binary masks into a single binary mask (logical OR)
    merged_mask = np.any(masks, axis=0)
    
    # Encode the merged mask back to RLE
    # merged_rle = mask_utils.encode(np.asfortranarray(merged_mask.astype(np.uint8)))
    
    return merged_mask



train_segment_data = []
with open(train_path) as f:
    train_data = json.load(f)
    
# print(train_data)
for qa in tqdm(train_data[:25000]):
    # print(qa)
    image_id = qa['image_id']
    # print(image_id)
    file_id = qa['id']
    file_id = file_id.replace('/','_')
    
    file_dict = {}
    question, answer, anat, type_qa = qa['question'], qa['grounded_answer'], qa['anat'], qa['type']
    # print(anat)
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
    if '-checkpoint' in image_id:
        image_id = image_id.replace('-checkpoint','')
    
    segment_file = os.path.join(segment_path, f'{image_id}.json')
    with open(segment_file) as f:
        segment_data = json.load(f)
    # print(segment_data['annotations'][0].keys())

    # print(segment_file)
    with open(os.path.join(train_output_dir,f'{file_id}.json'),'w') as f:
        json.dump(file_dict,f)
    # print(type_qa)
    if type_qa == 'adversarial_finding_open':
        exemplar_mask = segment_data['annotations'][0]['segmentation']
        exemplar_binary_mask = mask.decode(exemplar_mask)
        binary_mask = np.zeros(exemplar_binary_mask.shape)
        # print(binary_mask.shape)
        mask_path = os.path.join(train_mask_dir, f'{file_id}.npz')
        # np.save(mask_path, binary_mask)
        np.savez_compressed(mask_path, binary_mask = binary_mask)

#         pass
    elif type_qa == 'finding':
        # print('anat: ',anat)
        
        if ',' not in anat:
            if anat in ANATOMY_TO_ID_MAPPINGS:
                anat_id = ANATOMY_TO_ID_MAPPINGS[anat]
                for annotation in segment_data['annotations']:
                    # print(annotation)
                    if annotation['category_id'] == anat_id:
                        rle_mask = annotation['segmentation']
                        binary_mask = mask.decode(rle_mask)
                        # print(binary_mask.shape)
                        mask_path = os.path.join(train_mask_dir, f'{file_id}.npz')
                        # np.save(mask_path, binary_mask)
                        np.savez_compressed(mask_path, binary_mask = binary_mask)
        else:
            print('anat: ',anat)
            anatomies = anat.split(',')
            anatomies = [anat.strip() for anat in anatomies]
            rle_list = []
            for anat in anatomies:
                for annotation in segment_data['annotations']:
                    # print(annotation)
                    if annotation['category_id'] == anat_id:
                        rle_mask = annotation['segmentation']
                        rle_list.append(rle_mask)
                        # break
            binary_mask = merge_rle_encodings(rle_list)
            mask_path = os.path.join(train_mask_dir, f'{file_id}.npz')
            # np.save(mask_path, binary_mask)
            np.savez_compressed(mask_path, binary_mask = binary_mask)
            # try:

            # print(anatomies)
            # print(binary_mask.shape)

    else:
        if anat in ANATOMY_TO_ID_MAPPINGS:
            anat_id = ANATOMY_TO_ID_MAPPINGS[anat]
            for annotation in segment_data['annotations']:
                # print(annotation)
                if annotation['category_id'] == anat_id:
                    rle_mask = annotation['segmentation']
                    binary_mask = mask.decode(rle_mask)
                    # print(binary_mask.shape)
                    mask_path = os.path.join(train_mask_dir, f'{file_id}.npz')
                    # np.save(mask_path, binary_mask)
                    np.savez_compressed(mask_path, binary_mask = binary_mask)
        else:
            if anat in MIMIC_MAPPINGS:
                print(anat)
                anatomies = MIMIC_MAPPINGS[anat]
                rle_list = []
                for anat in anatomies:
                    anat_id = ANATOMY_TO_ID_MAPPINGS[anat]

                    for annotation in segment_data['annotations']:
                        # print(annotation)
                        
                        if annotation['category_id'] == anat_id:
                            rle_mask = annotation['segmentation']
                            rle_list.append(rle_mask)
                            # break
                binary_mask = merge_rle_encodings(rle_list)
                mask_path = os.path.join(train_mask_dir, f'{file_id}.npz')
                # np.save(mask_path, binary_mask)
                np.savez_compressed(mask_path, binary_mask = binary_mask)

            # try:
            #     contours = measure.find_contours(binary_mask, level=0.5)[0]
            #     points = contours.tolist()
            #     file_dict['shapes'][0]['points'] = points
            #     with open(os.path.join(train_output_dir,f'{file_id}.json'),'w') as f:
            #         json.dump(file_dict,f)
            # except:
            #     pass
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

for qa in tqdm(test_data[:5000]):
    image_id = qa['image_id']
    print(image_id)
    file_id = qa['id']
    file_id = file_id.replace('/','_')
    
    file_dict = {}
    question, answer, anat, type_qa = qa['question'], qa['grounded_answer'], qa['anat'], qa['type']
    # print(anat)
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
    if '-checkpoint' in image_id:
        image_id = image_id.replace('-checkpoint','')
    
    segment_file = os.path.join(segment_path, f'{image_id}.json')
    with open(segment_file) as f:
        segment_data = json.load(f)
    # print(segment_data['annotations'][0].keys())

    # print(segment_file)
    with open(os.path.join(test_output_dir,f'{file_id}.json'),'w') as f:
        json.dump(file_dict,f)
    # print(type_qa)
    if type_qa == 'adversarial_finding_open':
        exemplar_mask = segment_data['annotations'][0]['segmentation']
        exemplar_binary_mask = mask.decode(exemplar_mask)
        binary_mask = np.zeros(exemplar_binary_mask.shape)
        # print(binary_mask.shape)
        mask_path = os.path.join(test_mask_dir, f'{file_id}.npz')
        # np.save(mask_path, binary_mask)
        np.savez_compressed(mask_path, binary_mask = binary_mask)

#         pass
    elif type_qa == 'finding':
        # print('anat: ',anat)
        
        if ',' not in anat:
            if anat in ANATOMY_TO_ID_MAPPINGS:
                anat_id = ANATOMY_TO_ID_MAPPINGS[anat]
                for annotation in segment_data['annotations']:
                    # print(annotation)
                    if annotation['category_id'] == anat_id:
                        rle_mask = annotation['segmentation']
                        binary_mask = mask.decode(rle_mask)
                        # print(binary_mask.shape)
                        mask_path = os.path.join(test_mask_dir, f'{file_id}.npz')
                        # np.save(mask_path, binary_mask)
                        np.savez_compressed(mask_path, binary_mask = binary_mask)
        else:
            print('anat: ',anat)
            anatomies = anat.split(',')
            anatomies = [anat.strip() for anat in anatomies]
            rle_list = []
            for anat in anatomies:
                for annotation in segment_data['annotations']:
                    # print(annotation)
                    if annotation['category_id'] == anat_id:
                        rle_mask = annotation['segmentation']
                        rle_list.append(rle_mask)
                        # break
            binary_mask = merge_rle_encodings(rle_list)
            mask_path = os.path.join(test_mask_dir, f'{file_id}.npz')
            # np.save(mask_path, binary_mask)
            np.savez_compressed(mask_path, binary_mask = binary_mask)
            # try:

            # print(anatomies)
            # print(binary_mask.shape)

    else:
        if anat in ANATOMY_TO_ID_MAPPINGS:
            anat_id = ANATOMY_TO_ID_MAPPINGS[anat]
            for annotation in segment_data['annotations']:
                # print(annotation)
                if annotation['category_id'] == anat_id:
                    rle_mask = annotation['segmentation']
                    binary_mask = mask.decode(rle_mask)
                    # print(binary_mask.shape)
                    mask_path = os.path.join(test_mask_dir, f'{file_id}.npz')
                    # np.save(mask_path, binary_mask)
                    np.savez_compressed(mask_path, binary_mask = binary_mask)
        else:
            if anat in MIMIC_MAPPINGS:
                print(anat)
                anatomies = MIMIC_MAPPINGS[anat]
                rle_list = []
                for anat in anatomies:
                    anat_id = ANATOMY_TO_ID_MAPPINGS[anat]

                    for annotation in segment_data['annotations']:
                        # print(annotation)
                        
                        if annotation['category_id'] == anat_id:
                            rle_mask = annotation['segmentation']
                            rle_list.append(rle_mask)
                            # break
                binary_mask = merge_rle_encodings(rle_list)
                mask_path = os.path.join(test_mask_dir, f'{file_id}.npz')
                # np.save(mask_path, binary_mask)
                np.savez_compressed(mask_path, binary_mask = binary_mask)
