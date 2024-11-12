import os
from glob import glob

import cxas.visualize as cxas_vis


def visual_anatomy(class_names, img_path, label_path, img_size=512, cat=True, axis=1, do_store=False, out_dir="MIMIC_anatomy_visual"):
    cxas_vis.visualize_from_file(
        class_names=class_names,
        img_path=img_path,
        label_path=label_path,
        img_size=img_size,
        cat=cat,
        axis=axis,
        do_store=do_store,
        out_dir="MIMIC_anatomy_visual"
    )


def main():
    path_dir = "/home/data-center/medical-segmentation/hieu/2.0.0/files"
    path_root_save = "MIMIC_anatomy"

    path_masks = glob(os.path.join(path_root_save, "*/*/*/*.npy"))
    path_dir_test = "MIMIC_anatomy_visual"
    os.makedirs(path_dir_test, exist_ok=True)

    for mask_p in path_masks:
        path_relative = mask_p.rsplit("MIMIC_anatomy/", 1)[1].replace(".npy", ".jpg")
        path_image = os.path.join(path_dir, path_relative)
        visual_anatomy(
            class_names=['thoracic spine'],
            img_path=path_image,
            label_path=mask_p,
            do_store=True,
            out_dir=path_dir_test
        )


if __name__ == "__main__":
    main()
