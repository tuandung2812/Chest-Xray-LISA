import sys
sys.path.append(".")

import os

from dataloader_hub.dataloader_base.dataset_base_generic import GenericDatasetBase


class LISAVinDrGroundConvGenerationDataset(GenericDatasetBase):
    def __init__(
        self, 
        config
    ):
        super().__init__(config)

    def load_dataset(self):
        pass

    def extract_image_embeddings(self, data):
        pass

    def preprocess_image(self, data):
        pass

    def form_conversation(self, labels):
        pass

    def preprocess_text(self, data):
        pass
