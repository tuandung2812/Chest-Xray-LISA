import roboflow

rf = roboflow.Roboflow(api_key="OHpDZXVRCusSZDTGS9QU")

# get a workspace
workspace = rf.workspace("medical-aiml")

# Upload data set to a new/existing project
workspace.upload_dataset(
    "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/VinDr_MedLMM_COCO", # This is your dataset path
    project_name="anatomy-contouring-vindr",
    num_workers=10,
    project_license="MIT",
    project_type="instance-segmentation",
    batch_name=None,
    num_retries=0
)