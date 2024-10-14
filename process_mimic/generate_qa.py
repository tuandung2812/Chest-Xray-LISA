import openai
import json

# Set your OpenAI API key
import os

from openai import OpenAI


openai.api_key = 'INSERT KEY'
import pandas as pd
os.environ["OPENAI_API_KEY"] = 'INSERT KEY'
import time
client = OpenAI()

# Function to generate a chat completion using GPT-4
def generate_chat_completion(messages, model="gpt-4o-mini", max_tokens=1000):
    """
    Generates a response from OpenAI's GPT-4 model based on a series of messages.

    Args:
        messages (list): A list of messages for the conversation, where each message is a dictionary
                         with 'role' (either 'system', 'user', or 'assistant') and 'content'.
        model (str): The model to use for generating the response.
        max_tokens (int): The maximum number of tokens to generate in the response.

    Returns:
        dict: A dictionary containing the generated response and other metadata.
    """
# try:
    response =client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        n=1,
        stop=None
    )
    # print(response)
    # Extract the response content and return as a dictionary
    reply = response.choices[0].message.content
    result = {
        "input_messages": messages,
        "response": reply
    }
    return reply

    # except :
    #     print(f"An error occurred")
    #     return None

# Function to save the response to a JSON file
def save_to_json(data, filename="response.json"):
    """
    Saves data to a JSON file.

    Args:
        data (dict): The data to be saved in the JSON file.
        filename (str): The name of the file to save the data to.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Response saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving to JSON: {e}")

PROMPT_TEMPLATE = """
Given a list of diseases and the anatomies associated with them in the form of a dictionary. generate 5 questions of different types and give the correct answer. 
If the answer is yes for questions with anatomies, first segment it by giving answer instructions to segment the model with the <seg> token. here are some examples.
The list of anatomies available is ["left middle lung", 'right middle lung', 'left lower lung', 'right lower lung', 'right upper lung', left upper lung, "aorta","heart", "spine","mediastinum","stomach"]
The list of  diseases available is ["Enlarged Cardiom", "Cardiomegaly", "Lung Lesion", 'Lung Opacity","Edema", "Consolidation", "Pneumonia","Atelectasis", "Pneumothorax","Pleural Effusion", "Fracture"]

"yes_no_findings"
 - "Does the patient have cardiomegaly?"
 - "The location of the heart <seg>. The answer is Yes."


"anatomy_disease_yes_no_findings"
 - "Are there cardiomegaly in the heart of this patient?"
- "The location of the heart <seg>. No

 - "Does the aorta suffer from aortic enlargement?"
 - "The location of the aorta <seg>. Yes
 
"anatomy_disease_open_findings"
"Are there anything wrong with the lung?"
 - The location of the lung <seg>. The lung has pneumonia

Please constrict your answer to the above-mentioned types of questions. 
Also, please generate adversarial questions, which are question pairs that deliberately choose negative diseases with positive entities
For example, the predictions have a pneumonia at the right lower lung.  An adversarial question would be 
"Does the patient pneumonia at the left lower lung". 
"The location of the left lower lung <seg>. The answer is No". 
Choose the entity from the above list  ["left middle lung", 'right middle lung', 'left lower lung', 'right lower lung', 'right upper lung', left upper lung, "aorta","heart", "spine","mediastinum","stomach"]. 
If the true entity is a lung type (left lower lung), choose a negative entity another part of the lung (right upper lung)


Name this "adversarial_disease_anatomy"
Other type of adversarial questions is "adversarial_disease_only", where the question asks about a finding not present in the image. choose that finding from the list above
["Enlarged Cardiom", "Cardiomegaly", "Lung Lesion", 'Lung Opacity","Edema", "Consolidation", "Pneumonia","Atelectasis", "Pneumothorax","Pleural Effusion", "Fracture"]
"Does the patient have pneumonia"
"No"

Return question/answer pairs in the form of a list of dict. DO NOT DEVIATE FROM THIS FORMAT
[{type: "adversarial_disease_only", "question": "Does the patient have pneumonia","answer": "No"}
{type:"anatomy_disease_yes_no_findings"
 "question": "Are there cardiomegaly in the heart of this patient?","answer": "The location of the heart <seg>. No"}]
"""

# Example usage
if __name__ == "__main__":
    import json

    import os
    from tqdm import tqdm

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
        
        disease_mappings = json.dumps(disease_mappings)
        
        # print(disease_mappings, type(disease_mappings))
        # break
#         input_data = f'Word: {word} \n Definition: {definition}'
        input_data = f'\n Disease-anatomy mapping: {disease_mappings}'
        messages = [
                {"role": "system", 'content': PROMPT_TEMPLATE},
                {"role": "user", "content": input_data}
            ]
        response = generate_chat_completion(messages)
        cleaned_json_str = response.replace("json", "").strip('\n')
        print(cleaned_json_str)
        response_list = json.loads(cleaned_json_str.strip())
        print(response_list)
        answer_dict[image_name] = response_list
        with open('answers.json', 'w') as f:
            json.dump(answer_dict, f, indent=4)
