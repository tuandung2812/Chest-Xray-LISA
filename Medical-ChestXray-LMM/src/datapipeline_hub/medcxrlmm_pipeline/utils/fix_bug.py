import os
from tqdm import tqdm
from glob import glob
import json


def main():
    dirpath_seg = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR/VinDr_MedGLaMM/train_png_16bit"
    filepaths_seg = glob(os.path.join(dirpath_seg, "*.json"))
    for filepath_seg in tqdm(filepaths_seg):
        with open(filepath_seg, "r") as f:
            data = json.load(f)
            # print(data["annotations"])
            data["annotations"] = data["annotations"][0] 
        with open(filepath_seg, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
