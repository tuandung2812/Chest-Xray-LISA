import os
import glob
import json


def main():
    """"
    Merge seperated prediction files csv
    """
    dirpath_pred_yesno = "/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/probmed/probmed_yesno"
    filepaths_pred_yesno = glob.glob(os.path.join(dirpath_pred_yesno, "*.json"))

    for filepath_pred in filepaths_pred_yesno:
        filepath_pred_adversarial = filepath_pred.replace("/probmed_yesno/", "/probmed_yesno_adversarial/")
        assert os.path.exists(filepath_pred_adversarial), filepath_pred_adversarial
        data_normal = json.load(open(filepath_pred))
        data_adversarial = json.load(open(filepath_pred_adversarial))

        for filename_image, samples in data_adversarial.items():
            if len(samples) == 0:
                print(filename_image)
                continue
            if filename_image in data_normal.keys():
                data_normal[filename_image].extend(samples)
            else:
                data_normal[filename_image] = samples
        
        filepath_merged = filepath_pred.replace("/probmed_yesno/", "/probmed_yesno_merged/")
        with open(filepath_merged, "w") as f:
            json.dump(data_normal, f)


if __name__ == "__main__":
    main()
