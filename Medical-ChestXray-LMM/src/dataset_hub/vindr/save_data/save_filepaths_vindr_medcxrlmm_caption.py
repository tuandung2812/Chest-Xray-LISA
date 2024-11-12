import sys
sys.path.append(".")

import os
import glob
import csv

from dataset_hub.vindr.config_vindr.config_vindr_medcxrlmm import *


def main():
    filepath_labels = glob.glob(f"{DIRPATH_VINDR_MEDCXRLMM_CAPTION}/*.json")
    
    data_output = []
    for filepath_label in filepath_labels:
        filepath_label = os.path.join(DIRPATH_VINDR_MEDCXRLMM_CAPTION, os.path.basename(filepath_label))
        filepath_image = os.path.join(DIRPATH_IMAGES, os.path.basename(filepath_label).replace('.json', '.png'))
        assert os.path.exists(filepath_image), filepath_image
        assert os.path.exists(filepath_label), filepath_label
        data_output.append([os.path.basename(filepath_image), os.path.basename(filepath_label)])

    with open(FILEPATH_VINDR_MEDCXRLMM_CAPTION_FILEPATHS, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['filepath_image', 'filepath_label'])
        for row in data_output:
            writer.writerow(row)


if __name__ == '__main__':
    main()
