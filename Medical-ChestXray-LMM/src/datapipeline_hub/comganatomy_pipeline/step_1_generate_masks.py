import os
from tqdm import tqdm
from dotmap import DotMap

from cxas import CXAS

from prepare_data import load_data


def create_cxas_model():
    return CXAS(model_name='UNet_ResNet50_default', gpus='1')


def cxas_infer(cxas_model, path_image, out_path):
    cxas_model.process_file(
        filename=path_image,
        do_store=True, 
        output_directory=out_path,
        storage_type='npy',
    )


def main():
    cxas_model = create_cxas_model()

    data_meta = load_data()
    # path_dir = "/home/data-center/medical-segmentation/hieu"
    path_dir = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG"
    data_split = data_meta["train"]

    path_root_save = "MIMIC_anatomy"

    for sample in tqdm(data_split):
        sample = DotMap(sample)
        path_image = os.path.join(path_dir, sample.Path)
        path_dir_save = os.path.join(path_root_save, sample.Path.split("/")[-4], f"p{sample.subject_id}", sample.study_id)
        if not os.path.exists(path_dir_save):
            os.makedirs(path_dir_save)

        cxas_infer(cxas_model, path_image, path_dir_save)
        break


if __name__ == "__main__":
    main()
