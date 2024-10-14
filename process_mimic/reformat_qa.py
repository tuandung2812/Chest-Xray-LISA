import json

with open('vindr_qa_data_train.json') as f:
    train_data = json.load(f)
# print(train_data)

new_train_data = []
for pid in train_data:
    # print(pid)
    count = 0
    for qa in train_data[pid]:
        qa['id'] = f'{str(count)}_{pid}'
        qa['image_id'] = pid

        new_train_data.append(qa)
        count += 1

with open('vindr_qa_data_train_newformat.json','w') as f:
    json.dump(new_train_data,f)
    # train_data = json.load(f)
