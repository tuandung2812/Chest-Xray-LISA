import json
from langchain_openai import ChatOpenAI
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()

with open("/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/probmed_v2/vqa_vindr_unhealthy_image_normal_llavamed0.json", "r") as f:
    data = json.load(f)

flattened_data = []

for key, entries in data.items():
    for entry in entries:
        entry_data = {'id': key}
        entry_data.update(entry)
        flattened_data.append(entry_data)

df = pd.DataFrame(flattened_data)

llm = ChatOpenAI(model='gpt-4o-mini')  # consider gpt-4o

disease_list = ['aortic enlargement', 'atelectasis', 'calcification', 'cardiomegaly', 
              'consolidation', 'interstitial lung disease', 'infiltration', 'lung opacification', 'nodule/mass', 
              'other lesion', 'pleural effusion', 'pleural thickening', 
              'pneumothorax', 'pulmonary fibrosis', 'no finding']

def regex_abn(response):
    output_pattern = r'==OUTPUT==\s*\[([^]]+)\]'

    output_match = re.search(output_pattern, response)
    if output_match:
        output_list = [
            abnormality.strip().strip('"').strip("'") 
            for abnormality in output_match.group(1).split(',')
        ]
        return output_list
    return []

def extract_abnormalities(query):
    if query.lower().strip() == 'no':
        return []
    prompt = f"""You are an expert radiologist. You have to read this conclusion and determine which abnormalities are indicated as positive. Only consider the abnormalities that are given in this list:
{disease_list}.
Answer following this format:
==REASONING==
<give 1-2 concise sentences about your reasonings, is there any synnonym or inclusion from the query compared with the list? scarring can be considered as fibrosis, groundglass is a subset of lung opacity>
==OUTPUT==
<return all confirmed abnomalites from the query that belong to the list>
[abnormality 1, abnormality 2, ...]
_____
The rules:
- Keep the ==REASONING== and ==OUTPUT== marker for easy extraction. 
- Remove all content in <>.
- Make sure to return a list of strings.
- only consider the abnormalities from the list.
- return no finding only when it is very explicit, any uncertainty or abnormalities cannot result in no finding, even though they are not from the list.
Follow the rule and the list or you will be fired.

The conclusion is: {query}"""
    response = llm.invoke(prompt).content
    extracted = regex_abn(response)
    extracted = [c for c in extracted if c != '']
    for c in extracted:
        if c not in disease_list:
            print('retry', extracted)
            return extract_abnormalities(query)
    return extracted

if __name__ == '__main__':
    import argparse
    from fastprogress import progress_bar
    
    parser = argparse.ArgumentParser(description="Input VQA model name")
    parser.add_argument('--name', type=str, required=True, 
                        help='The vqa model name to choose')

    args = parser.parse_args()
    model = args.name

    print(f"VQA model: {args.name}")
    
    MAX_TRIAL = 2
    # vqa_models = [
    #     'chexagenthf', 'llavamed0'
    # ]

    with open(f'{model}_gpt.tsv', 'w') as f:
        f.writelines(f"id\tpred\tresult\n")
    bar = progress_bar(df.iterrows(), total=len(df))
    for i,c in bar:
        if c["question_type"] != "abnormality":
            continue
        query = c['response']
        for _ in range(MAX_TRIAL):
            try:
                result = extract_abnormalities(query)
                break
            except Exception as e:
                print(e)
                continue
        result = ",".join(result)
        bar.comment = f"{result}"
        
        with open(f'{model}_gpt.tsv', 'a') as f:
            f.writelines(f"{c.id}\t{query}\t{result}\n")