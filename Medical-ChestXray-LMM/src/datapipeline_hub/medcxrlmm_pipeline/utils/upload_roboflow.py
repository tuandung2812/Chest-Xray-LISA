# import roboflow

# rf = roboflow.Roboflow(api_key="OHpDZXVRCusSZDTGS9QU")

# # get a workspace
# print(rf.workspace())
# workspace = rf.workspace("https://app.roboflow.com/medical-aiml")


# # Upload data set to a new/existing project
# workspace.upload_dataset(
#     "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/VinDr_MedLMM_COCO", # This is your dataset path
#     "anatomy-contouring-vindr", # This will either create or get a dataset with the given ID
#     num_workers=10,
#     project_license="CC By 4.0",
#     project_type="instance-segmentation",
#     batch_name=None,
#     num_retries=0
# )

from roboflow import Roboflow

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="OHpDZXVRCusSZDTGS9QU")

# Retrieve your current workspace and project name
print(rf.workspace())

# Specify the project for upload
# let's you have a project at https://app.roboflow.com/my-workspace/my-project
workspaceId = 'medical-aiml'
projectId = 'anatomy-contouring-vindr'
project = rf.workspace(workspaceId).project(projectId)

# Upload the image to your project
from glob import glob
from tqdm import tqdm
filepaths_image = glob("/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/VinDr_MedLMM_COCO/*.png")
for filepath_image in tqdm(filepaths_image):
    project.upload(
        image_path=filepath_image,
        batch_name="batch-sdk",
        split="train",
        num_retry_uploads=3,
    )
