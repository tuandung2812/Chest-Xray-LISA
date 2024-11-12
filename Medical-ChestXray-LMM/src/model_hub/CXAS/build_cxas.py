import json

from .modeling import CXAS, CXAS_ImageEncoder, CXAS_ImageDecoder_Head
from .load_ckpt_cxas import load_ckpt_origin


def build_cxas_unet_resnet(
) -> CXAS:
    paxray_labels = json.load(open('/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/src/model_hub/CXAS/paxray_labels.json'))
    id2label_dict = paxray_labels['label_dict']
    
    cxas = CXAS(
        image_encoder=CXAS_ImageEncoder(
            network_name="UNet_ResNet50_default"
        ),
        image_decoder=CXAS_ImageDecoder_Head(
            in_channels=2048,
            ngf=128,
            num_classes=len(id2label_dict.keys()),
            norm='batch' if 1 > 1 else 'instance'
        )
    )
    cxas.eval()
    cxas = load_ckpt_origin(cxas)
    return cxas


cxas_model_registry = {
    "cxas_unet_resnet50": build_cxas_unet_resnet,
}


def test():
    import torch

    model = cxas_model_registry["cxas_unet_resnet50"]()
    data = torch.rand(size=(1, 3, 512, 512))
    output = model(data)
    print(output.shape)


if __name__ == "__main__":
    test()
