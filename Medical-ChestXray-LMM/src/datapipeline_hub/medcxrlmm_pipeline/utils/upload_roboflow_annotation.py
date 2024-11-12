import glob
from roboflow import Roboflow

# Initialize Roboflow client
rf = Roboflow(api_key="OHpDZXVRCusSZDTGS9QU")

# Directory path and file extension for images
dir_name = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/VinDr_MedLMM_COCO"
file_extension_type = ".png"

# Annotation file path and format (e.g., .coco.json)
annotation_filename = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/VinDr_MedLMM_COCO/_annotations.coco.json"

# Get the upload project from Roboflow workspace
project = rf.workspace().project("anatomy-contouring-vindr")

# Upload images
image_glob = glob.glob(dir_name + '/*' + file_extension_type)
for image_path in image_glob:
    print(project.single_upload(
        image_path=image_path,
        annotation_path=annotation_filename,
        # optional parameters:
        # annotation_labelmap=labelmap_path,
        split='train',
        # num_retry_uploads=0,
        batch_name='batch-sdk',
        # tag_names=['tag1', 'tag2'],
        # is_prediction=False,
    ))
    break