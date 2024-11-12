import pandas as pd

from dataset_hub.vindr.config_vindr.config_vindr_0 import *


def main():
    dirpath_images = "/mnt/12T/02_duong/data-center/VinDr/train_png_16bit"
    filepath_labels = "/mnt/12T/02_duong/data-center/VinDr/train.csv"

    df = pd.read_csv(filepath_labels)
    unique_ids = df['class_id'].unique()
    print(unique_ids)


if __name__ == '__main__':
    main()
