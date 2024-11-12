import json

def load_chestx(path):
    with open(path, 'r') as f:
        data0 = json.load(f)
    data = data0["mapping"]
    output = []
    for mapping in data:
        filepath_image = list(mapping.values())[0]
        filepath_image_rel = filepath_image.replace("/mnt/12T/01_hieu/VLM/data/xray14_resized_224x224/cxr/", "")
        output.append(
            {
                "Path": filepath_image_rel
            }
        )
    return output
