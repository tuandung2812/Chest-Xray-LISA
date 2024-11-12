import os
import gdown

import torch

model_urls = {
    'UNet_ResNet50_default': 'https://drive.google.com/file/d/1Y9zubvMzkYHoAqz-NvV6vniH5FKAF2iV/view?usp=drive_link'
}


def load_ckpt_origin(model):
    model_name = "UNet_ResNet50_default"
    download_weights(model_name)
    model = load_weights(model, model_name, map_location = 'cpu')
    return model


def download_weights(model_name):
    if "CXAS_PATH" in os.environ:
        store_path = os.path.join(os.environ['CXAS_PATH'],'.cxas')
    else:
        store_path = os.path.join(os.environ['HOME'],'.cxas')
    os.makedirs(os.path.join(store_path, 'weights/'), exist_ok=True)
    out_path = os.path.join(store_path, 'weights/{}'.format(model_name+'.pth'))
    if os.path.isfile(out_path):
        return
    else:
        gdown.download(model_urls[model_name], out_path, quiet=False, fuzzy=True)
        return
    

def load_weights(model, model_name:str, map_location:str='cuda:0'):
    if "CXAS_PATH" in os.environ:
        store_path = os.path.join(os.environ['CXAS_PATH'],'.cxas')
    else:
        store_path = os.path.join(os.environ['HOME'],'.cxas')
    out_path = os.path.join(store_path, 'weights/{}'.format(model_name+'.pth'))
    assert os.path.isfile(out_path)
    
    checkpoint = torch.load(out_path, map_location=map_location)
    # import pdb; pdb.set_trace()
    if 'module' in list(checkpoint['model'].keys())[0] :
        for i in list(checkpoint['model'].keys()):
            checkpoint['model'][i[len('module.'):]] = checkpoint['model'].pop(i)

    state_dict0 = checkpoint['model']
    state_dict_customized = {}
    for k, v in state_dict0.items():
        if "backbone.backbone." in k:
            k_new = k.replace("backbone.backbone.", "image_encoder.backbone.")
        elif "head." in k:
            k_new = k.replace("head.", "image_decoder.")
        state_dict_customized[k_new] = v

    model.load_state_dict(state_dict_customized, strict = True)
    return model