import json



with open('process_mimic/vindr_qa_data_train_newformat.json') as f:
    train_data = json.load(f)


import random
random.seed(42)
# print(type(train_data))
all_types = []

type_qa_dict = {}
for data in train_data:
    if data['type'] not in all_types:
        all_types.append(data['type'])
    if data['type'] not in type_qa_dict:
        type_qa_dict[data['type']] = []
    else:
        type_qa_dict[data['type']].append(data)

new_train_data = []
# print(all_types)
for key in type_qa_dict:
    sample_data = random.sample(type_qa_dict[key],5000)
    new_train_data.extend(sample_data)
# print(len(new_data))
random.shuffle(new_train_data)
with open('process_mimic/vindr_qa_train_resampled.json','w') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)

    
with open('process_mimic/vindr_qa_data_test_newformat.json') as f:
    train_data = json.load(f)

# with open('process_mimic/vindr_qa_data_test_newformat.json') as f:
#     test_data = json.load(f)

import random
random.seed(42)
# print(type(train_data))
all_types = []

type_qa_dict = {}
for data in train_data:
    # print(all_types)
    # if data['type'] not in all_types
    if data['type'] not in all_types:
        all_types.append(data['type'])
    if data['type'] not in type_qa_dict:
        type_qa_dict[data['type']] = []
    else:
        type_qa_dict[data['type']].append(data)

new_train_data = []
# print(all_types)
for key in type_qa_dict:
    print(key, len(type_qa_dict[key]))
    if key == 'adversarial_anatomy_disease_yes_no':
        sample_data = random.sample(type_qa_dict[key], 1000)
    elif key == 'adversarial_anatomy_open':
        sample_data = random.sample(type_qa_dict[key], 1000)
    elif key == 'adversarial_finding_open':
        sample_data = random.sample(type_qa_dict[key], 1000)

    else:
        sample_data = type_qa_dict[key]
    new_train_data.extend(sample_data)
# print(len(new_data))
random.shuffle(new_train_data)
with open('process_mimic/vindr_qa_test_resampled.json','w') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)

    
    
    

import random
random.seed(42)
# print(type(train_data))
all_types = []
with open('process_mimic/mimic_qa_data_train_newformat.json') as f:
    train_data = json.load(f)

type_qa_dict = {}
for data in train_data:
    # print(all_types)
    # if data['type'] not in all_types
    if data['type'] not in all_types:
        all_types.append(data['type'])
    if data['type'] not in type_qa_dict:
        type_qa_dict[data['type']] = []
    else:
        type_qa_dict[data['type']].append(data)

new_train_data = []
# print(all_types)
for key in type_qa_dict:
    print(key, len(type_qa_dict[key]))
    if key == 'adversarial_anatomy_disease_yes_no':
        sample_data = random.sample(type_qa_dict[key], 1000)
    elif key == 'adversarial_anatomy_open':
        sample_data = random.sample(type_qa_dict[key], 1000)
    elif key == 'adversarial_finding_open':
        sample_data = random.sample(type_qa_dict[key], 1000)

    else:
        sample_data = type_qa_dict[key]
    new_train_data.extend(sample_data)
# print(len(new_data))
random.shuffle(new_train_data)
with open('process_mimic/mimic_qa_train_resampled.json','w') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)

    
with open('process_mimic/mimic_qa_data_test_newformat.json') as f:
    train_data = json.load(f)

import random
random.seed(42)
# print(type(train_data))
all_types = []

type_qa_dict = {}
for data in train_data:
    # if data['type'] not in all_types
    if data['type'] not in all_types:
        all_types.append(data['type'])
    if data['type'] not in type_qa_dict:
        type_qa_dict[data['type']] = []
    else:
        type_qa_dict[data['type']].append(data)

new_train_data = []
# print(all_types)
for key in type_qa_dict:
    print(key, len(type_qa_dict[key]))
    if key == 'adversarial_anatomy_disease_yes_no':
        sample_data = random.sample(type_qa_dict[key], 1000)
    elif key == 'adversarial_anatomy_open':
        sample_data = random.sample(type_qa_dict[key], 1000)
    elif key == 'adversarial_finding_open':
        sample_data = random.sample(type_qa_dict[key], 1000)

    else:
        sample_data = type_qa_dict[key]
    new_train_data.extend(sample_data)
# print(len(new_data))
random.shuffle(new_train_data)
with open('process_mimic/mimic_qa_test_resampled.json','w') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)
