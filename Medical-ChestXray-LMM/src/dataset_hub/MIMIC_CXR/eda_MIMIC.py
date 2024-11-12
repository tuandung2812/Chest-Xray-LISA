import sys
sys.path.append("/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/src")
from datapipeline_hub.COMG_Anatomy_pipeline.prepare_data import load_data


def get_disease_images(data_meta, disease_name="mild pulmonary edema"):
    import os
    import shutil
    
    path_dir = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG"
    path_dest = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/tmp/MIMIC/diseases"

    i = 1
    for data in data_meta["train"]:
        findings = data["findings"]
        impression = data["impression"]

        if disease_name in findings or disease_name in impression:
            path_file = os.path.join(path_dir, data["Path"])
            # shutil.copy(path_file, path_dest)

            print(path_file)
            print("#" * 10 + "FINDINGS")
            print(findings.replace("\n", ""))
            print("#" * 10 + "IMPRESSIONS")
            print(impression.replace("\n", ""))
            print("#" * 20)
            print()


        
        i +=1
        if i == 100:
            break


def main():
    data_meta = load_data(path_data="COMG_Anatomy_pipeline/MIMIC_meta_data.json")

    get_disease_images(data_meta)


if __name__ == "__main__":
    main()
