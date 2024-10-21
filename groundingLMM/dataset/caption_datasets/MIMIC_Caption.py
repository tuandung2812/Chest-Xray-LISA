import os
import json
import torch
from torch.utils.data import Dataset
import openai
import random

class RadiologyReportDataset(Dataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024

    def __init__(self, dataset_dir, openai_api_key, epoch_samples=10000, image_size=224, validation=False, random_sampling=True):
        """
        Initialize the dataset by loading the JSON data and setting up OpenAI API.

        Args:
        - dataset_dir (str): Path to the dataset directory containing merged_data.json.
        - openai_api_key (str): OpenAI API key for accessing GPT-4.
        - epoch_samples (int): Number of samples per epoch.
        - image_size (int): Size to which images are resized.
        - validation (bool): Whether this is a validation dataset.
        - random_sampling (bool): Whether to use random sampling of data.
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.openai_api_key = openai_api_key
        self.epoch_samples = epoch_samples
        self.validation = validation
        self.random_sampling = random_sampling

        # Load data from JSON file
        json_path = os.path.join(self.dataset_dir, 'merged_data.json')
        with open(json_path, 'r') as file:
            self.data = json.load(file)

        self.image_ids = list(self.data.keys())

        # Initialize OpenAI API
        openai.api_key = self.openai_api_key
        self.begin_str = "Based on the radiology report provided, answer the following questions:\n"
        print('\033[92m' + "----CAP-{}: Radiology Report dataset initialized----".format("Val" if validation else "Train") + '\033[0m')

    def __len__(self):
        """
        Return the total number of image IDs in the dataset.
        """
        return len(self.image_ids)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize and pad the image tensor.
        
        Args:
        - x (torch.Tensor): The image tensor.

        Returns:
        - torch.Tensor: Processed image tensor.
        """
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = torch.nn.functional.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, report, question):
        """
        Create conversation prompts based on the report and question.

        Args:
        - report (str): The radiology report.
        - question (str): The question to answer.

        Returns:
        - list, list: A list of questions and a list of generated conversations.
        """
        conversations = []
        questions = []

        input_text = f"Report: {report}\nQuestion: {question}\nAnswer:"
        questions.append(question)
        answer = self.generate_answer(report, question)

        prompt = self.begin_str + input_text + answer
        conversations.append(prompt)
        return questions, conversations

    def generate_answer(self, report, question):
        """
        Generate an answer using the GPT-4 API based on the report and question.

        Args:
        - report (str): The radiology report.
        - question (str): The question to be answered.

        Returns:
        - str: Generated answer.
        """
        prompt = f"Report: {report}\nQuestion: {question}\nAnswer:"
        
        try:
            response = clients.chat.completion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise and accurate responses based on radiology reports."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                n=1,
                stop=None,
                temperature=0.7
            )
            answer = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            # print(f"Error generating answer with GPT-4 API: {e}")
            answer = "No answers"
        
        return answer

    def __getitem__(self, idx):
        """
        Get the data for a given index.

        Args:
        - idx (int): Index to retrieve data.

        Returns:
        - tuple: Contains image path, global_enc_image, grounding_enc_image, bounding boxes,
                 conversations, masks, label, resized image, questions, and selected labels.
        """
        image_id = self.image_ids[idx] if (self.validation or not self.random_sampling) else random.choice(self.image_ids)
        report = self.data[image_id]['report']
        questions = self.data[image_id]['questions']

        # Select a random question
        question = random.choice(questions)
        questions, conversations = self.create_conversations(report, question)
        selected_labels = conversations

        # In this dataset, we are not using images, so we'll mock those parts as None or empty tensors
        image_path = None  # No actual image path
        global_enc_image = None
        grounding_enc_image = None
        image_resize = None
        bboxes = None
        masks = torch.rand(0, *([self.IMG_SIZE, self.IMG_SIZE]))
        label = None

        assert len(conversations) == 1

        return (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
                questions, selected_labels)

# Example usage
if __name__ == "__main__":
    openai_api_key = ""  # Replace with your actual API key
    dataset = RadiologyReportDataset(dataset_dir='p', openai_api_key=openai_api_key)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for data in dataloader:
        print(data)
