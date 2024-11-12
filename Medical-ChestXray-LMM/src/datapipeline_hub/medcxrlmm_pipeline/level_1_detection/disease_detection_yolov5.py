import os
from tqdm import tqdm
from dotmap import DotMap

import cv2
import random

from datapipeline_hub.comganatomy_pipeline.prepare_data import load_data
from datapipeline_hub.datasets.vindr.load_data import load_dataset_vin
from model_hub.yolo_v5.predict_yolo_v5 import YOLO_v5_Handler
from model_hub.yolo_v5.utils.plots import plot_one_box

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def config_mimic(split="train", context=None):
    data_meta = load_data(path_data="../COMG_Anatomy_pipeline/MIMIC_meta_data.json")
    data_split = data_meta[split]

    path_images_dir = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG"

    path_fix = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR"
    if context.is_mirror:
        ckpt_id = context.checkpoint.split("/")[-1].split(".")[0] + "_mirror"
    else:
        ckpt_id = context.checkpoint.split("/")[-1].split(".")[0]
    path_root_save = os.path.join(path_fix, "MIMIC_disease_YOLO_v5", ckpt_id)
    path_root_save_visual = os.path.join(path_fix, "MIMIC_disease_YOLO_v5_visual", ckpt_id)

    return data_split, path_images_dir, path_root_save, path_root_save_visual


def config_vindr(split="train", context=None):
    data_meta = load_dataset_vin(path="/mnt/12T/02_duong/data-center/VinDr/train.csv")
    path_images_dir = f"/mnt/12T/02_duong/data-center/VinDr/{split}_png_16bit"

    path_fix = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR"
    if context.is_mirror:
        ckpt_id = context.checkpoint.split("/")[-1].split(".")[0] + "_mirror"
    else:
        ckpt_id = context.checkpoint.split("/")[-1].split(".")[0]
    path_root_save = os.path.join(path_fix, "VinDr_disease_YOLO_v5", f"{split}_png_16bit", ckpt_id)
    path_root_save_visual = os.path.join(path_fix, "VinDr_disease_YOLO_v5_visual", f"{split}_png_16bit", ckpt_id)

    return data_meta, path_images_dir, path_root_save, path_root_save_visual


def main():
    context = {
        "model_id": "yolo_v5",
        "checkpoint": "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/opensources/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/models_inference/yolo_best/best_fold_0_mAP_0.383_0.184.pt",
        # "checkpoint": "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/opensources/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/models_inference/yolo_best/best_fold_1_mAP_0.395_0.182.pt",
        # "checkpoint": "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/opensources/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/models_inference/yolo_best/best_fold_2_mAP_0.415_0.196.pt",
        # "checkpoint": "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/opensources/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/models_inference/yolo_best/best_fold_3_mAP_0.382_0.183.pt",
        # "checkpoint": "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/opensources/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/models_inference/yolo_best/best_fold_4_mAP_0.409_0.189.pt",
        "device": "cuda",
        "image_size": 640,
        "conf_thres": 0.01,
        "iou_thres": 0.25,
        "classes": None,
        "agnostic_nms": None,
        "is_mirror": False
    }
    context = DotMap(context)
    print(context.checkpoint)

    # data_split, path_images_dir, path_root_save, path_root_save_visual = config_mimic(context=context)
    data_split, path_images_dir, path_root_save, path_root_save_visual = config_vindr(context=context)
    yolo_v5_infer = YOLO_v5_Handler(
        context=context
    )
    disease_names = yolo_v5_infer.model.names
    print(disease_names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in disease_names]    

    for sample in tqdm(data_split):
        sample = DotMap(sample)
        path_image = os.path.join(path_images_dir, sample.Path)

        data = {
            "path_image": path_image
        }
        output = yolo_v5_infer.handle(data)
        if output["status"] == "Error":
            continue

        # save coordination
        data_coor = output["data_coor"]
        # path_dir_save = os.path.join(path_root_save, sample.Path.split("/")[-4], f"p{sample.subject_id}", sample.study_id)
        path_dir_save = path_root_save
        os.makedirs(path_dir_save, exist_ok=True)
        filename_out = path_image.split("/")[-1].split(".")[0] + ".txt"
        filepath_out = os.path.join(path_dir_save, filename_out)
        with open(filepath_out, "w") as f:
            for det in data_coor:
                disease_class, *xywh, conf = det
                f.write(f"{disease_class} {conf} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")

        # save visual image
        data_visual = output["data_visual"]
        if data_visual is not None:
            image0 = output["image0"]
            # path_dir_visual_save = os.path.join(path_root_save_visual, sample.Path.split("/")[-4], f"p{sample.subject_id}", sample.study_id)
            path_dir_visual_save = path_root_save_visual
            os.makedirs(path_dir_visual_save, exist_ok=True)
            filename_visual_out = path_image.split("/")[-1].split(".")[0] + ".jpg"
            filename_visual_out = os.path.join(path_dir_visual_save, filename_visual_out)

            for *xyxy, conf, disease_class in data_visual:
                if conf < 0.5:
                    continue
                label = f'{disease_names[int(disease_class)]} {conf:.2f}'
                plot_one_box(xyxy, image0, label=label, color=colors[int(disease_class)], line_thickness=3)
            cv2.imwrite(filename_visual_out, image0)


if __name__ == "__main__":
    main()
