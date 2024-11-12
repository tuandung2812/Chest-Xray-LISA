import os
from tqdm import tqdm
from dotmap import DotMap
import json

import sys
sys.path.append("/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/src")
from datapipeline_hub.comganatomy_pipeline.prepare_data import load_data
from model_hub.torchxrayvision import XRV_Handler
from model_hub.cxas import resize_to_numpy, export_prediction_as_numpy, visualize_from_file


def main():
    data_meta = load_data(path_data="../COMG_Anatomy_pipeline/MIMIC_meta_data.json")
    # path_dir = "/home/data-center/medical-segmentation/hieu"
    path_dir = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG"
    data_split = data_meta["train"][:20]

    context={
            "model_id": "xrv_pspnet",
            "device": "cuda",
            "threshold": 0.5
        }
    context = DotMap(context)
    xrv_infer = XRV_Handler(
        context=context
    )
    paxray_labels = json.load(open('/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/src/model_hub/CXAS/paxray_labels.json'))
    id2label_dict = paxray_labels['label_dict']

    xrv_label_meta = ['clavicle left', 'clavicle right', 'scapula left', 'scapula right',
                   'left lung', 'right lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
                   'heart', 'aorta', 'Facies Diaphragmatica', 'mediastinum',  'weasand', 'spine']

    path_root_save = "MIMIC_anatomy_XRV"
    path_root_save_visual = "MIMIC_anatomy_XRV_visual"

    for sample in tqdm(data_split):
        sample = DotMap(sample)
        path_image = os.path.join(path_dir, sample.Path)
        path_dir_save = os.path.join(path_root_save, sample.Path.split("/")[-4], f"p{sample.subject_id}", sample.study_id)
        if not os.path.exists(path_dir_save):
            os.makedirs(path_dir_save)
        data = {
            "image_path": path_image
        }
        output = xrv_infer.handle(data)
        # pred_resize = resize_to_numpy(
        #     segmentation=output["post_data"],
        #     file_size=output["file_size"]
        # )

        pred_resize = segmentation=output["post_data"]
        
        # filename_out = path_image.split("/")[-1].split(".")[0] + ".npy"
        # filepath_out = os.path.join(path_dir_save, filename_out)
        # export_prediction_as_numpy(pred_resize, filepath_out)

        path_dir_visual_save = os.path.join(path_root_save_visual, sample.Path.split("/")[-4], f"p{sample.subject_id}", sample.study_id)
        if not os.path.exists(path_dir_visual_save):
            os.makedirs(path_dir_visual_save)

        # for _, name_anatomy in id2label_dict.items():
        #     for xrv_label in xrv_label_meta:
        #         if xrv_label not in name_anatomy:
        #             pass
        #         else:
        #             print(1)
        #             visualize_from_file(
        #                 class_names=[name_anatomy], 
        #                 img_path=path_image, 
        #                 label_path=filepath_out, 
        #                 img_size=512, 
        #                 cat=True, 
        #                 axis=1, 
        #                 do_store=True,
        #                 out_dir=path_dir_visual_save,
        #             )
        
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        img = np.array(Image.open(path_image).convert(mode="RGB"))
        print(pred_resize.shape)
        plt.figure(figsize = (26,5))
        plt.subplot(1, len(xrv_label_meta) + 1, 1)
        plt.imshow(img[-1], cmap='gray')
        for i in range(len(xrv_label_meta)):
            plt.subplot(1, len(xrv_label_meta) + 1, i+2)
            plt.imshow(pred_resize[0, i])
            plt.title(xrv_label_meta[i])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(path_dir_visual_save + "/pred.png")


if __name__ == "__main__":
    main()
