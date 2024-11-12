import os
from tqdm import tqdm
from dotmap import DotMap

from datapipeline_hub.comganatomy_pipeline.prepare_data import load_data
from datasets.vindr.load_data import load_dataset_vin
from model_hub.cxas import (
    CXAS_Handler, 
    resize_to_numpy, 
    export_prediction_as_numpy, 
    export_prediction_as_json,
    visualize_from_file,
    id2label_dict
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def config_mimic(split="train"):
    data_meta = load_data(path_data="../../COMG_Anatomy_pipeline/MIMIC_meta_data.json")
    data_split = data_meta[split]

    path_images_dir = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG"

    path_fix = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR"
    path_root_save = os.path.join(path_fix, "MIMIC_anatomy_CXAS_2808")
    path_root_save_visual = os.path.join(path_fix, "MIMIC_anatomy_CXAS_visual")

    os.makedirs(path_root_save, exist_ok=True)
    os.makedirs(path_root_save_visual, exist_ok=True)

    return data_split, path_images_dir, path_root_save, path_root_save_visual


def config_vindr(split="train"):
    data_meta = load_dataset_vin(path="/mnt/12T/02_duong/data-center/VinDr/train.csv")
    path_images_dir = f"/mnt/12T/02_duong/data-center/VinDr/{split}_png_16bit"

    path_fix = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR"
    path_root_save = os.path.join(path_fix, "VinDr_anatomy_CXAS", f"{split}_png_16bit")
    path_root_save_visual = os.path.join(path_fix, "VinDr_anatomy_CXAS_visual", f"{split}_png_16bit")

    os.makedirs(path_root_save, exist_ok=True)
    os.makedirs(path_root_save_visual, exist_ok=True)

    return data_meta, path_images_dir, path_root_save, path_root_save_visual


def config_chestx(split="train"):
    from datapipeline_hub.datasets.chestx.load_chestx import load_chestx

    data_meta = load_chestx(path=f"/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/src/data_pipeline_hub/datasets/chestx/mapping_chestx_{split}.json")
    path_images_dir = f"/mnt/12T/01_hieu/VLM/data/xray14_resized_224x224/cxr"

    path_root_save = os.path.join("/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/ChestX/ChestX_CXAS")
    path_root_save_visual = os.path.join("/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/ChestX_CXAS_visual")

    os.makedirs(path_root_save, exist_ok=True)
    os.makedirs(path_root_save_visual, exist_ok=True)

    return data_meta, path_images_dir, path_root_save, path_root_save_visual


def main(dataset_name="chestx", split="test"):
    if dataset_name == "mimic":
        data_split, path_images_dir, path_root_save, path_root_save_visual = config_mimic(split)
    elif dataset_name == "vindr":
        data_split, path_images_dir, path_root_save, path_root_save_visual = config_vindr(split)
    elif dataset_name == "chestx":
        data_split, path_images_dir, path_root_save, path_root_save_visual = config_chestx(split)
    else:
        raise ValueError

    context={
            "model_id": "cxas_unet_resnet50",
            "device": "cuda",
            "base_size": 512,
            "threshold": 0.5
        }
    context = DotMap(context)
    cxas_infer = CXAS_Handler(
        context=context
    )

    for sample in tqdm(data_split):
        sample = DotMap(sample)
        path_image = os.path.join(path_images_dir, sample.Path)
        data = {
            "path_image": path_image
        }
        output = cxas_infer.handle(data)
        if output["status"] == "Error":
            continue
        pred_resize = resize_to_numpy(
            segmentation=output["post_data"],
            file_size=output["orig_size"]
        )

        # save mask mimic
        if dataset_name == "mimic":
            path_dir_save = os.path.join(path_root_save, sample.Path.split("/")[-4], f"p{sample.subject_id}", sample.study_id)
        elif dataset_name == "vindr":
            path_dir_save = path_root_save
        elif dataset_name == "chestx":
            path_dir_save = os.path.join(path_root_save, os.path.dirname(sample.Path))
        os.makedirs(path_dir_save, exist_ok=True)
        
        filename_out = path_image.split("/")[-1].split(".")[0] + ".json"
        filepath_out = os.path.join(path_dir_save, filename_out)
        export_prediction_as_json(
            mask=pred_resize,
            out_path=filepath_out,
            base_ann_id=1,
            filepath=sample.Path
        )   

        # filename_out = path_image.split("/")[-1].split(".")[0] + ".npy"
        # filepath_out = os.path.join(path_dir_save, filename_out)
        # export_prediction_as_numpy(
        #     mask=pred_resize,
        #     out_path=filepath_out,
        # )
        
        # if dataset_name == "mimic":
        #     path_dir_visual_save = os.path.join(path_root_save_visual, sample.Path.split("/")[-4], f"p{sample.subject_id}", sample.study_id)
        # elif dataset_name == "vindr":
        #     path_dir_visual_save = path_root_save_visual
        # elif dataset_name == "chestx":
        #     path_dir_visual_save = os.path.join(path_root_save_visual, os.path.dirname(sample.Path))
        # os.makedirs(path_dir_visual_save, exist_ok=True)

        # for _, name_anatomy in id2label_dict.items():
        #     visualize_from_file(
        #         class_names=[name_anatomy], 
        #         img_path=path_image, 
        #         label_path=filepath_out, 
        #         img_size=512, 
        #         cat=True, 
        #         axis=1, 
        #         do_store=True,
        #         out_dir=path_dir_visual_save,
        #     )


if __name__ == "__main__":
    main()
