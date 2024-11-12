import os
import json

import shutil

path = '/home/user01/aiotlab/dung_paper/groundingLMM/LISAMed/dataset/VinDr/vindr_zip/'

sample_path = '/home/user01/aiotlab/dung_paper/groundingLMM/LISAMed/dataset/VinDr/vindr_sampled/'

if not os.path.exists(sample_path):
    os.makedirs(sample_path)

chosen_files=   [
    "4_e62c07fde352cc658af3f989fe0b546f",
    "4_e521b0e2fca887f754c05d26a9783103",
    "4_ee7cc34733914fd5924eb5d75a27fba3",
    "6_c3abe21f7e07452e6760cdc2cab95296",
    "5_f59ccb89d776a68b79292bef810333ac",
    "5_f51c1a48919f8b36116ed4aa799dcb23",
    "5_f7dd175ca0962155ec39311b997a5288",
    "5_824920c963d82a8c0a2a593177ad5b93",
    "7_1e96e84ac07738cd2f019b1c1d1aac5c",
    "2_96bd06e0d91d69607522a2f8cab22550",
    "2_95b1441a7db0a7fa07549ab95ddb4ef9",
    "11_0c31081e8ada2990bcbef0f12ea60b07"
]

for file_id in chosen_files:
    img_id = file_id.split('_')[-1]
    image_path = os.path.join(path,'jpg','all',img_id) + '.jpg'
    dest_image_path = os.path.join(sample_path,'jpg' ,img_id) + '.jpg'

    if not os.path.exists(dest_image_path):
        os.makedirs(dest_image_path)
    shutil.copy(image_path, dest_image_path)
    
    annotation_path = os.path.join(path,'annotation_resampled','test',file_id) + '.json'
    dest_annotation_path = os.path.join(sample_path,'annotation_resampled',file_id) + '.json'

    if not os.path.exists(dest_annotation_path):
        os.makedirs(dest_annotation_path)
    shutil.copy(annotation_path, dest_annotation_path)

    mask_path = os.path.join(path,'mask_resampled','test',file_id) + '.npz'
    dest_mask_path = os.path.join(sample_path,'mask_resampled',file_id) + '.npz'

    if not os.path.exists(dest_mask_path):
        os.makedirs(dest_mask_path)
    shutil.copy(mask_path, dest_mask_path)

    print(image_path)
    print(os.path.exists(mask_path))
    print(os.path.exists(annotation_path))
    print(os.path.exists(image_path))
