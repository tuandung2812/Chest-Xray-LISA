import os

def get_txt_files(folder_path):
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files

# Sử dụng
folder_path = "/home/user01/aiotlab/dung_paper/groundingLMM/dataset/mimic_processed/MIMIC_MedGLaMM_caption/p10/"
txt_files = get_txt_files(folder_path)
# print(txt_files)
# for file in txt_files:
#     print(file)

import json
data_dict = {}
for file_path in txt_files:
    with open(file_path, 'r') as file:
        file_content = file.read()
    # print(file_path)
    image_name = '/'.join(file_path.split('/')[-3:]).replace('.txt','.jpg')
    image_id = image_name.replace('.jpg','')
    print(image_name, image_id)
    # Chuyển đổi từ chuỗi JSON thành từ điển Python
    data_dict = json.loads(file_content)
    # print(data_dict['impression'])
    for impression in data_dict['impression']:
        print(impression)
        disease_name = impression['disease']['name']
        print(disease_name)
        anatomies = []
        seg_masks = []
        height = 1500
        width = 2250
        for anatomy in impression['anatomies']:
            anatomies.append(anatomy['name_anatomy'])
            seg_masks.append(anatomy['anatomy_mask']['segmentation']['counts'])
        print(anatomies)
    data_dict
    # break
