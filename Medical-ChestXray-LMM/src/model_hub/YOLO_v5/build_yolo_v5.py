from .load_ckpt_yolo_v5 import load_ckpt_origin


def build_yolo_v5(
    checkpoint
):
    yolo_v5 = load_ckpt_origin(checkpoint)
    yolo_v5.eval()
    return yolo_v5


yolo_v5_registry = {
    "yolo_v5": build_yolo_v5,
}


def test():
    import torch

    path_model = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/opensources/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/models_inference/yolo_best/best_fold_0_mAP_0.383_0.184.pt"
     
    model = yolo_v5_registry["yolo_v5"](
        checkpoint=path_model
    )
    print(model.names)

    data = torch.rand(size=(1, 3, 640, 640))
    output = model(data)
    print(output[0].shape)
    for i in output[1]:
        print(i.shape)


if __name__ == "__main__":
    test()
