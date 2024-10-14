import os
import json
from tqdm import tqdm
def get_txt_files(folder_path):
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files

def get_token_before_seg(caption, seg_token="<seg>"):
    # Tokenize the caption (split by spaces)
    tokens = caption.split()
    
    # Find the position of the <seg> token
    token_positives = []
    for i, token in enumerate(tokens):
        if token == seg_token:
            # If the <seg> token is found, check if there's a token before it
            if i > 0:
                # Store the word before <seg> and the <seg> itself
                token_positives = [tokens[i-1], tokens[i]]  # [word_before_seg, seg_token]
                # Also return the positions
                positions = [i-1, i]  # [position_of_word_before_seg, position_of_seg]
            break
    
    return token_positives, positions
# Sử dụng
folder_path = "/home/user01/aiotlab/dung_paper/groundingLMM/dataset/mimic_processed/MIMIC_MedGLaMM_caption/p10/"
txt_files = get_txt_files(folder_path)
# print(txt_files)
# for file in txt_files:
#     print(file)
with open('answers.json') as f:
    qa_data = json.load(f)
# print(qa_data)
glamm_lists = []
for image_name in tqdm(qa_data):
    print(image_name)
    image_id = image_name.replace('.jpg','')
    
    data_file = os.path.join( "/home/user01/aiotlab/dung_paper/groundingLMM/dataset/mimic_processed/MIMIC_MedGLaMM_caption/p10/", image_id + '.txt')
    # print(data_file)
    with open(data_file) as f:
        data = json.load(f)
    # print(data)
    height = 1500
    width = 2250
    anatomy_seg_mappings = {}
    for impression in data['impression']:
        # print(impression)
        disease_name = impression['disease']['name']
        # print(disease_name)
        anatomies = []
        seg_masks = []
        for anatomy in impression['anatomies']:
            anatomies.append(anatomy['name_anatomy'])
            seg_masks.append(anatomy['anatomy_mask']['segmentation']['counts'])
            anatomy_seg_mappings[anatomy['name_anatomy'].lower()] = anatomy['anatomy_mask']['segmentation']['counts']
        # print(anatomies)
        # print(anatomy_seg_mappings)
    for qa_pair in qa_data[image_name]:
        question, answer = qa_pair['question'], qa_pair['answer']
        glamm_dict = {'file_name': image_name, 'height': height,'width': width, 'image_id':image_id,'question':question,'caption': answer,'groundings':{}}
        anatomies = [anatomy.lower() for anatomy in anatomy_seg_mappings.keys() if anatomy.lower() in answer]
        # print(answer)
        if len(anatomies) == 0:
            if '<seg>' in answer:
                parts = answer.split('<seg>')
                # Get the word before "<seg>" and "<seg>"
                before_seg = parts[0].strip().split()[-1]  # Last word before <seg>
                # print(before_seg)
                word = before_seg + ' ' + "<seg>"
                # print(word)
                start_position = answer.find(word)
                end_position = start_position + len(word) - 1
                
                # print(start_position, end_position)
                rle = "b==V^1d0_O<F9G:F9H7I?A9G7I6K5K6J6J8H8G6K4L3L4N2M3M3M3M3M4J5I7E<G8F;H7K6K4K6K6J5L4L4L4L3M3M3M3M3M3M3L4M2O2M3N2M4L4M3L4M3M2M4M3M3L5L4K5K5L2N2M3N3M6I7J2M4M4L3M3M3M3L3M4M4K6K3M2N3M2N3M2N3M3M2N2M3N2N3L3N3M2N2N2M3N2N2N3L3N3L4M2N3M2N3M2N2N2N2M3N2N2N2O1N2N1O2N3M2N2N2N2N1O2N2O0O2N3M2O1N2N2O1N2N2O1N3N1N2O1N101N2N101N2N101N2O0O2O1N2O0O2N101N1N3N1M4L3L4L5L4M2N3L3N3M2N3M2N3N1N3N1O2M2O1N2O2M2N2O1O2N1O1O2O0O2O0O2N1O1O2N1O1N2O1O2M2O1N2O1O1N3N1N2O1O1O1O1O1O1N2O1O1O1O1O100O1O1O1O1O2N100O1O1O1O1O2N100O2N100O2N101N1O2O0O2O0O2O0O1O101N1O100O2O0O1O101N1O1O2N1O101N1O1N3N1N2O1N3N1O1N2O1N2O1N2O1O2N1O1N2O1O1O1O1O1N200O1O2N1O1O1O1O100O1N3N1O1O1O1O1O2N1N2O1O1O1O1O2N1O1O1O1O1O1O1O1O1O1O1O1O1O100O1O100O1O100O1O1O10O0100O1O10000O10000O10000O10001N2O0O2O1O0O2O001N101N101N2O0O2O1N2N101N2N2N1O2N1O2O0O2M2O2N1O2N1O2N1O1O2M2O1O2M2O1O2N1O1O1O2N1O1O1O2N1O1O2N1O101N1O1O1O2N1O1O2N1O1O2N1O1O2O0O1O2O0O1O101N100O2O0O1O2O0O2N100O2N1O101N1O100O2O0O100O100O100O100O100O100O2O0O100O100O10001O0O2O000O2O001N101O1N2O001N101N101N101N101N101O0O2O1O0O2O000O2O0O2O000O101O000O101O0O10001O0O101O0O10001N10000O2O0O10000O101N10000O10000O2O0O100O10000O101O0O100O100O100O10001N100O100O100O100O100O100O2N100O10000O100O100O100O1000001N1000000O10001N10000O10001N10000O10001N10000O1000001O000O1000001O0O10000O10001O0O100O101N10000O101O0O100000001O000O1000001O000000000O2O00000000001N10000000001O000000000O1000001O00000000000O100000001O0000000000000000000000001O00000000000000O100000000000001O00000O100000000000000O1000000O10000O10000O10000O10000O100O10000O100O100O100O001O1O1O1O1O1O10O1000O1000000001N10001O00000O2O0000001O00001O00001O0O2O001O00001O00001O00001N100000001O0000001O00001O000000001O000000001O01O00000001O000000000000001O00000000000000000000000000000000000000O10000000001O00000O1000001O000O101O001O000O2O00000O2O0000000O2O0000001O0O101O00001N10001O0O101O00001N101O000O2O00001N10001N1000001N10001N1000001N101O000O2O001O0O2O0O10000O2O000O101O000O101O000O2O00001O00001O00001O0000001O001N10001O00001O00001O0000001O00001O001O0000001O00001O00001O00001O001O00001O00001O00001O001O001O001O00001O00001O001O00001N101O001O001O001O001O001N101O001O001O001N10001O001O0O101O001O000O2O00001O0O101O001N101O001O1O1O001N2O00001N10001O0O101O001N10001N101O0O2O000O2O0O2O000O2O0O2O1O001N2O1O1N101O1O0O2O001N101O0O2O1O0O2O0O2O1N101O1N101N101O1N101N2O1N2O1N2N101N2N1O2N101N101O0O2O1N101O1N101O1O001N2O1O001O1O00100O1O1O001O10O01O010O001O00010O0000001O01O01O000010O01O00000010O01O00001O0010O01O001O0010O0001O00001O000010O0001O00001O001O00001O001O001O001O001O001O001O0O2O001O001O001O001O1O001O1O1N101O1O001O0O2O1O001O001O001N101O001O1O1O1O1N101O1O1O1O0O2O1O001O1N2O1O001N2O001N2O001N101O1N101O1N2O1O0O2O1O1N2O1O1N3N1N2O0O2O1N101N2O0O2O1N2O0O2O1N2O0O2O1N2N2O0O2O0O2O1N101N101N2O001N2N101N1O2O0O2O1N101N2O0O2O0O2O1O0O2O1N2O1N101O0O2O1N101N2O1N101N2O0O2N101N2N2O1N2N2O1N2N2O0O2N2O0O2N2N1O2O1N2N1O2N2N1O2N1O2N1O2N2N2N2N3M2N2N2N2N2N2M3N2N1O2N2N2N2N2N2N2O1N2N2O1N2O1N2N3N1N2O1O2M2O1O1N2O1O1N2O1N2O1N2N2O1N2O1N3N1N2O1N2O1O1N2O1O1N2O2M2O1O1N3N2N1N2O2M2O1N3N1N3N1N3N2M3N2M4M3L3N2M4M3L3N2M2O2M2N3N1N2N2O1N2N2N2N2N2N2N2N2N2N1O2N2N1O2N2N2N3M2N3L4M3L4M2M3N2N1N3N2M2N3M3M3M3N3L3N2N3M2M4K5K4M3M4M2N3M2N3L4L3M4M2M4L4M2M4L5K4L5K4K5K5L3M4M4K5K5K4J7J5K6K4K5L5J6I6K6H7J7I7I7G8J7I7G<D<_O`0E;C>C?A;Db0^OV1RN`bN5bjo?"
                glamm_dict['groundings'][word] = {'token_positives': [{0:start_position}, {1: end_position}], 'rle_masks': [{'size': [ {0: 1500}, {1:2250}], 'counts':rle}]}
        else:
            for anatomy in anatomies:
                print(anatomy)
                rle  = anatomy_seg_mappings[anatomy]
                word = anatomy + ' ' + '<seg>'
                start_position = answer.find(word)
                end_position = start_position + len(word) - 1
                glamm_dict['groundings'][word] = {'token_positives': [{0:start_position}, {1: end_position}], 'rle_masks': [{'size': [ {0: 1500}, {1:2250}], 'counts':rle}]}
        
        glamm_lists.append(glamm_dict)
        
        with open('glamm_qa.json','w') as f:
            json.dump(glamm_lists, f, indent=  4)
        # print(glamm_lists)
                                
                                                                                                  
                                                                                                                
                                                                                                                        


        # print(anatomies)
# import json
# data_dict = {}
# for file_path in txt_files:
#     with open(file_path, 'r') as file:
#         file_content = file.read()
#     # print(file_path)
#     image_name = '/'.join(file_path.split('/')[-3:]).replace('.txt','.jpg')
#     image_id = image_name.replace('.jpg','')
#     print(image_name, image_id)
#     # Chuyển đổi từ chuỗi JSON thành từ điển Python
#     data_dict = json.loads(file_content)
#     # print(data_dict['impression'])
#     for impression in data_dict['impression']:
#         print(impression)
#         disease_name = impression['disease']['name']
#         print(disease_name)
#         anatomies = []
#         seg_masks = []
#         height = 1500
#         width = 2250
#         for anatomy in impression['anatomies']:
#             anatomies.append(anatomy['name_anatomy'])
#             seg_masks.append(anatomy['anatomy_mask']['segmentation']['counts'])
#         print(anatomies)
#     data_dict
#     # break
