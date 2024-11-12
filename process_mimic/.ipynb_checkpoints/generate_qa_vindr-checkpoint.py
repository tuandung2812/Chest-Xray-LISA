import openai
import json

# Set your OpenAI API key
import os

from openai import OpenAI
from collections import Counter

import pandas as pd
import time
import random


QUESTION_TEMPLATES = {
'anatomy_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",
    
    # "List all abnormalities in {anatomy}?",
    # "What abnormalities are present in {anatomy}?",
    # "Does {anatomy} suffer from any abnormalities?",
    # "List all abnormalities in {anatomy}?",
    # "What abnormalities are present in {anatomy}?",
    # "What abnormalities does {anatomy} suffer from?",

],
'anatomy_open': [
    "List all abnormalities in {anatomy}?",
    "What abnormalities are present in {anatomy}?",
    "What abnormalities does {anatomy} suffer from?",
],

'adversarial_easy_at_anatomy_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",
],

'adversarial_easy_anatomy_at_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",
],

'adversarial_hard_at_anatomy_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",
],

'adversarial_hard_anatomy_at_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",],

'adversarial_anatomy_open': [
    # "Are there any observations at {anatomy}?",
    # "Does the patient's {anatomy} have any abnormalities?",
    # "Is there anything wrong with {anatomy}?",
    "List all abnormalities in {anatomy}?",
    "What abnormalities are present in {anatomy}?",
    "What abnormalities does {anatomy} suffer from?",

    # "Does {anatomy} suffer from any abnormalities?",
],

'adversarial_finding_open': [
    # "Are there any observations at {anatomy}?",
    # "Does the patient's {anatomy} have any abnormalities?",
    # "Is there anything wrong with {anatomy}?",
    "Does this patient have {disease}?",
    "Is {disease} present in this image?",

    # "Does {anatomy} suffer from any abnormalities?",
],

    
'finding_open': [
    # "Are there any observations at {anatomy}?",
    # "Does the patient's {anatomy} have any abnormalities?",
    # "Is there anything wrong with {anatomy}?",
    "Does this patient have {disease}?",
    "Is {disease} present in this image?",

    # "Does {anatomy} suffer from any abnormalities?",
],



}

ANSWER_TEMPLATES = {
'anatomy_disease_yes_no' : "The location of {anatomy} is at <seg>. The answer is Yes",

'anatomy_open': "The location of {anatomy} is at <seg>. It suffers from ",

'adversarial_easy_at_anatomy_disease_yes_no' : "The location of {anatomy} is at <seg>. The answer is No",

'adversarial_easy_anatomy_at_disease_yes_no' : "The location of {anatomy} is at <seg>. The answer is No",

'adversarial_hard_at_anatomy_disease_yes_no' :"The location of {anatomy} is at <seg>. The answer is No",

'adversarial_hard_anatomy_at_disease_yes_no' : "The location of {anatomy} is at <seg>. The answer is No",

'adversarial_anatomy_open':  "The location of {anatomy} is at <seg>. It has no abnormalities",

}
# a
# Example usage
if __name__ == "__main__":
    import json
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    import os
    from tqdm import tqdm

    def get_txt_files(folder_path):
        txt_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    txt_files.append(os.path.join(root, file))
        return txt_files

    # Sử dụng
    folder_path = "/home/user01/aiotlab/dung_paper/groundingLMM/LISAMed/dataset/VinDr/VinDr_MedGLaMM_caption/train_png_16bit"
    txt_files = get_txt_files(folder_path)
    # print(txt_files)
    # for file in txt_files:
    #     print(file)
    
    all_diseases = []
    all_anatomies = []
    all_mappings = []
    possible_disease_anat_mappings = {}
    possible_anat_disease_mappings = {}
    healthy_patients = []
    non_healthy_patients = []
    answer_dict ={}
    import json
    for file_path in tqdm(txt_files):
        with open(file_path, 'r') as file:
            file_content = file.read()
        data_dict = json.loads(file_content)

        image_name = '/'.join(file_path.split('/')[-3:]).replace('.txt','.jpg')
        image_id = image_name.replace('.jpg','')

        #     # print(data)
        disease_mappings = {}
        for impression in data_dict['impression']:
            # print(impression)
            disease_name = impression['disease']['name']
            disease_mappings[disease_name] = []
            for anatomy in impression['anatomies']:
                disease_mappings[disease_name].append(anatomy['name_anatomy'])
        
        # disease_mappings = json.dumps(disease_mappings)
        if disease_mappings:
            for disease, anat_list in disease_mappings.items():
                # print(disease, anat_list)
                if disease not in possible_disease_anat_mappings:
                    possible_disease_anat_mappings[disease] = []
                all_diseases.append(disease)
                for anat in anat_list:
                    if anat not in possible_anat_disease_mappings:
                        possible_anat_disease_mappings[anat] = []

                    all_anatomies.append(anat)
                    all_mappings.append(f'{disease} - {anat}')
                    if anat not in possible_disease_anat_mappings[disease]:
                        possible_disease_anat_mappings[disease].append(anat)
                        
                    if disease not in possible_anat_disease_mappings[anat]:
                        possible_anat_disease_mappings[anat].append(disease)
        
        all_diseases
    
    count_diseases = dict(Counter(all_diseases))
    count_anat = dict(Counter(all_anatomies))
    count_mappings = dict(Counter(all_mappings))
    
    all_diseases, all_anats = list(count_diseases.keys()), list(count_anat.keys())
    
    
    questions_data = {}
    for file_path in tqdm(txt_files):
        with open(file_path, 'r') as file:
            file_content = file.read()
        data_dict = json.loads(file_content)

        image_name = '/'.join(file_path.split('/')[-1:]).replace('.json','.jpg')
        image_id = image_name.replace('.jpg','')
        # print(image_id)
        questions_data[image_id] = []

        #     # print(data)
        disease_mappings = {}
        
        for impression in data_dict['impression']:
            # print(impression)
            disease_name = impression['disease']['name']
            disease_mappings[disease_name] = []
            for anatomy in impression['anatomies']:
                disease_mappings[disease_name].append(anatomy['name_anatomy'])
        
        if disease_mappings:
            non_healthy_patients.append(image_id)

            # print(disease_mappings)
            anat_mappings = {}

            # Lặp qua từng key-value trong original_dict
            for abnormality, anatomies in  disease_mappings.items():
                for anatomy in anatomies:
                    # Nếu giải phẫu chưa có trong reversed_dict, khởi tạo danh sách trống
                    if anatomy not in anat_mappings:
                        anat_mappings[anatomy] = []
                    # Thêm bất thường vào danh sách tương ứng
                    anat_mappings[anatomy].append(abnormality)
            # print(anat_mappings)
            
            positive_diseases = random.sample(list(disease_mappings.keys()), min(2, len(list(disease_mappings.keys()))))
            
            for disease in positive_diseases:
                pos_anat = random.choice(disease_mappings[disease])

                pos_anat_question = random.choice(QUESTION_TEMPLATES['anatomy_disease_yes_no']).format(disease=disease, anatomy=pos_anat)
                # print("Hard: ", hard_adv_anat_question)
                grounded_answer = ANSWER_TEMPLATES['anatomy_disease_yes_no'].format(anatomy=pos_anat)
                questions_data[image_id].append({'type': 'anatomy_disease_yes_no',
                                       'finegrained_type':  'anatomy_disease_yes_no', 
                                       'question':pos_anat_question,
                                       "answer": "Yes",
                                       "grounded_answer":grounded_answer, 'anat': pos_anat})
                
                anatomy_open_question = random.choice(QUESTION_TEMPLATES['anatomy_open']).format(anatomy=pos_anat)
                # print(anatomy_open_question)
                grounded_answer = ANSWER_TEMPLATES['anatomy_open'].format(anatomy=pos_anat)
                answer = "It suffers from"
                for i in range(len(anat_mappings[pos_anat])):
                    disease = anat_mappings[pos_anat][i]
                    if len(anat_mappings[pos_anat]) == 1:
                        grounded_answer += f' {disease}.'
                        answer +=  f" {disease}."
                    elif i != len(anat_mappings[pos_anat]) - 1:
                        grounded_answer += f' {disease},'
                        answer +=  f" {disease},"
                    else:
                        grounded_answer += f' {disease}.'
                        answer +=  f" {disease}."

                # print(grounded_answer)
                # grounded_answer = ANSWER_TEMPLATES['anatomy_open'].format(anatomy=pos_anat, disease=disease)
                questions_data[image_id].append({'type': 'anatomy_open',
                                       'finegrained_type':  'anatomy_open', 
                                       'question':anatomy_open_question,
                                       "answer":answer,
                                       "grounded_answer":grounded_answer, 'anat': pos_anat})

                # print(disease, pos_anat)
                neg_anats = list(set(all_anats) - set(disease_mappings[disease]))
                hard_neg_anats = list(set(possible_disease_anat_mappings[disease]) & set(neg_anats))
                easy_neg_anats =list(set(neg_anats) - set(hard_neg_anats))
                
                if hard_neg_anats:
                    hard_anat = random.choice(hard_neg_anats)
                    hard_adv_anat_question = random.choice(QUESTION_TEMPLATES['adversarial_hard_at_anatomy_disease_yes_no']).format(disease=disease, anatomy=hard_anat)
                    # print("Hard: ", hard_adv_anat_question)
                    grounded_answer_hard = ANSWER_TEMPLATES['adversarial_hard_at_anatomy_disease_yes_no'].format(anatomy=hard_anat)
                    questions_data[image_id].append({'type': 'adversarial_anatomy_disease_yes_no',
                                           'finegrained_type':  'adversarial_hard_at_anatomy_disease_yes_no', 
                                           'question':hard_adv_anat_question,
                                           "answer": "No",
                                           "grounded_answer":grounded_answer_hard, 'anat':hard_anat })

                if easy_neg_anats:
                    easy_anat = random.choice(easy_neg_anats)
                    easy_adv_anat_question = random.choice(QUESTION_TEMPLATES['adversarial_easy_at_anatomy_disease_yes_no']).format(disease=disease, anatomy=easy_anat)
                    # print("Hard: ", hard_adv_anat_question)
                    grounded_answer_easy = ANSWER_TEMPLATES['adversarial_easy_at_anatomy_disease_yes_no'].format(anatomy=easy_anat)
                    questions_data[image_id].append({'type': 'adversarial_anatomy_disease_yes_no',
                                           'finegrained_type':  'adversarial_easy_at_anatomy_disease_yes_no', 
                                           'question':easy_adv_anat_question,
                                           "answer": "No",
                                           "grounded_answer":grounded_answer_easy,'anat': easy_anat })


                    
                
            positive_anats = random.sample(list(anat_mappings.keys()), min(2, len(list(anat_mappings.keys()))))
            for anat in positive_anats:
                pos_disease = random.choice(anat_mappings[anat])
                neg_diseases =  list(set(all_diseases) - set(anat_mappings[anat]))
                hard_neg_diseases = list(set(possible_anat_disease_mappings[anat]) & set(neg_diseases))
                easy_neg_diseases =list(set(neg_diseases) - set(hard_neg_diseases))
                
                if hard_neg_diseases:
                    hard_disease = random.choice(hard_neg_diseases)
                    hard_adv_disease_question = random.choice(QUESTION_TEMPLATES['adversarial_hard_anatomy_at_disease_yes_no']).format(disease=hard_disease, anatomy=anat)
                    grounded_answer_hard = ANSWER_TEMPLATES['adversarial_hard_anatomy_at_disease_yes_no'].format(anatomy=anat)

                    questions_data[image_id].append({'type': 'adversarial_anatomy_disease_yes_no',
                        'finegrained_type': 'adversarial_hard_anatomy_at_disease_yes_no',
                        'question': hard_adv_disease_question,
                        'answer': 'No',
                        'grounded_answer': grounded_answer_hard, 'anat': anat})

                if easy_neg_diseases:
                    easy_disease = random.choice(easy_neg_diseases)
                    easy_adv_disease_question = random.choice(QUESTION_TEMPLATES['adversarial_easy_anatomy_at_disease_yes_no']).format(disease=easy_disease, anatomy=anat)
                    grounded_answer_easy = ANSWER_TEMPLATES['adversarial_easy_anatomy_at_disease_yes_no'].format(anatomy=anat)

                    questions_data[image_id].append({'type': 'adversarial_anatomy_disease_yes_no',
                        'finegrained_type': 'adversarial_easy_anatomy_at_disease_yes_no',
                        'question': easy_adv_disease_question,
                        'answer': 'No',
                        'grounded_answer': grounded_answer_easy, 'anat': anat})
                
                disease = random.choice(list(disease_mappings.keys()))
                disease_open_question = random.choice(QUESTION_TEMPLATES['finding_open']).format(disease=disease)
                anats = disease_mappings[disease]
                # print(anatomy_open_question)
                # grounded_answer = ANSWER_TEMPLATES['disea_open'].format(anatomy=pos_anat)
                grounded_answer = 'The location of '
                answer = "Yes"
                for i in range(len(anats)):
                    anat = anats[i]
                    # if i != len(anats) - 1:
                    grounded_answer += f' {anat},'
                        # answer +=  f" {disease},"
                    # else:
                    #     grounded_answer += f' {anat}.'
                        # answer +=  f" {disease}."
                grounded_answer += ' is at [SEG]. The answer is Yes'
                # print(grounded_answer)
                # grounded_answer = ANSWER_TEMPLATES['anatomy_open'].format(anatomy=pos_anat, disease=disease)
                questions_data[image_id].append({'type': 'finding',
                                       'finegrained_type':  'finding_open', 
                                       'question':disease_open_question,
                                       "answer":answer,
                                       "grounded_answer":grounded_answer, 'anat': ", ".join(anats) })

                # print(hard_neg_anats, easy_neg_anats)
                # print(neg_anats)
                
                # print(disease_mappings.keys(),neg_anats, len(neg_anats))
                # print(disease, pos_anat)
            # print(positive_diseases)
            pass

        # If len == 0, generate adversarial only
        else:
            healthy_patients.append(image_id)
            possible_types = ['adversarial_easy_at_anatomy_disease_yes_no' , 'adversarial_easy_anatomy_at_disease_yes_no',
                             'adversarial_anatomy_open']
            # Generate yes-no adversarial questions 
            
            # 'adversarial_easy_at_anatomy_disease_yes_no', 'adversarial_hard_at_anatomy_disease_yes_no' 
            random_disease = random.choice(all_diseases)
            possible_anats = possible_disease_anat_mappings[random_disease]
            hard_anat = random.choice(possible_anats)
            hard_adv_anat_question = random.choice(QUESTION_TEMPLATES['adversarial_hard_at_anatomy_disease_yes_no']).format(disease=random_disease, anatomy=hard_anat)
            # print("Hard: ", hard_adv_anat_question)
            grounded_answer_hard = ANSWER_TEMPLATES['adversarial_hard_at_anatomy_disease_yes_no'].format(anatomy=hard_anat)
            questions_data[image_id].append({'type': 'adversarial_anatomy_disease_yes_no',
                                   'finegrained_type':  'adversarial_hard_at_anatomy_disease_yes_no', 
                                   'question':hard_adv_anat_question,
                                   "answer": "No",
                                   "grounded_answer":grounded_answer_hard, 'anat': hard_anat })

            if len(possible_anats) != len(all_anats):
                impossible_anats = list(set(all_anats) - set(possible_anats))
                easy_anat = random.choice(impossible_anats)

                easy_adv_anat_question = random.choice(QUESTION_TEMPLATES['adversarial_easy_at_anatomy_disease_yes_no']).format(disease=random_disease, anatomy=easy_anat)
                # print("Easy " ,easy_adv_anat_question)
                grounded_answer_easy = ANSWER_TEMPLATES['adversarial_hard_at_anatomy_disease_yes_no'].format(anatomy=easy_anat)

                questions_data[image_id].append({'type': 'adversarial_anatomy_disease_yes_no',
                                       'finegrained_type':  'adversarial_easy_at_anatomy_disease_yes_no', 
                                       'question':easy_adv_anat_question,
                                       "answer": "No",
                                       "grounded_answer": grounded_answer_easy, 'anat': easy_anat})
            
            
            # 'adversarial_anatomy_open'
            anat = random.choice(all_anats)
            open_question = random.choice(QUESTION_TEMPLATES['adversarial_anatomy_open']).format(anatomy=anat)
            # print("open: ", open_question)
            grounded_open_answer = ANSWER_TEMPLATES['adversarial_anatomy_open'].format(anatomy=anat)
            
            questions_data[image_id].append({'type': 'adversarial_anatomy_open',
                                       'finegrained_type':  'adversarial_anatomy_open', 
                                       'question':open_question,
                                       "answer": "It suffers from no abnormalities",
                                       "grounded_answer": grounded_open_answer, 'anat': anat})

            disease = random.choice(all_diseases)
            open_question = random.choice(QUESTION_TEMPLATES['adversarial_finding_open']).format(disease=disease)
            # print("open: ", open_question)
            grounded_open_answer = "No"
            
            questions_data[image_id].append({'type': 'adversarial_finding_open',
                                       'finegrained_type':  'adversarial_finding_open', 
                                       'question':open_question,
                                       "answer": "No",
                                       "grounded_answer": grounded_open_answer, 'anat': None})

    # print(questions_data)      
    file_name = 'vindr_qa_data.json'
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(questions_data, json_file, ensure_ascii=False, indent=4)
    
    
    split_ratio = 0.8
    train_healthy_size = int(len(healthy_patients) * split_ratio)
    train_non_healthy_size = int(len(non_healthy_patients) * split_ratio)
    
    train_healthy_patients = random.sample(healthy_patients, train_healthy_size)
    test_healthy_patients = [key for key in healthy_patients if key not in train_healthy_patients]

    train_non_healthy_patients = random.sample(non_healthy_patients, train_non_healthy_size)
    test_non_healthy_patients = [key for key in non_healthy_patients if key not in train_non_healthy_patients]
    
    train_patients = train_healthy_patients + train_non_healthy_patients
    test_patients = test_healthy_patients + test_non_healthy_patients
    random.shuffle(train_patients)
    random.shuffle(test_patients)
    
    print(len(train_patients), len(test_patients))

    train_data = {key: questions_data[key] for key in train_patients}
    test_data = {key: questions_data[key] for key in test_patients}

    with open('process_mimic/vindr_qa_data_train.json', 'w', encoding='utf-8') as json_file:
        json.dump(train_data, json_file, ensure_ascii=False, indent=4)

    with open('process_mimic/vindr_qa_data_test.json', 'w', encoding='utf-8') as json_file:
        json.dump(test_data, json_file, ensure_ascii=False, indent=4)


        # print(questions_data)
            # break

            # print(random_disease, possible_anats, impossible_anats)
        # disease_mappings = json.dumps(disease_mappings)

        
    print(count_diseases)
    print(count_anat)
    print(count_mappings)
    # with open(count_diseases
    print(possible_disease_anat_mappings)
    print(possible_anat_disease_mappings)

        # print(disease_mappings)
        # print(disease_mappings, type(disease_mappings))
        # break
#         input_data = f'Word: {word} \n Definition: {definition}'
        # input_data = f'\n Disease-anatomy mapping: {disease_mappings}'
        # messages = [
        #         {"role": "system", 'content': PROMPT_TEMPLATE},
        #         {"role": "user", "content": input_data}
        #     ]
        # response = generate_chat_completion(messages)
        # cleaned_json_str = response.replace("json", "").strip('\n')
        # print(cleaned_json_str)
        # response_list = json.loads(cleaned_json_str.strip())
        # print(response_list)
        # answer_dict[image_name] = response_list
        # with open('answers.json', 'w') as f:
        #     json.dump(answer_dict, f, indent=4)
