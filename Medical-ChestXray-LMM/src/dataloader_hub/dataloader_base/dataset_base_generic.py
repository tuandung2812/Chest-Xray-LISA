import torch


class GenericDatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        dataset_split,
        visual_encoder,
    ):
        self.initialize_dataset_config(config, dataset_split, visual_encoder)
        self.dataset = self.load_dataset()

    def initialize_dataset_config(self, config, dataset_split, visual_encoder):
        raise NotImplementedError

    def load_dataset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset) 
    
    def process_image(self, data):
        raise NotImplementedError
    
    def form_conversation(self, data):
        raise NotImplementedError
    
    def process_text(self, data):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
