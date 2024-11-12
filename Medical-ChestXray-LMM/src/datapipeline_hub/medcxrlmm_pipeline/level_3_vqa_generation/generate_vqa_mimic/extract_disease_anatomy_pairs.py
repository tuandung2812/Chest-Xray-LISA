import os
import json
import tqdm

import numpy as np


anatomy_chexmask = {
    "left upper lung": 0,
    "left middle lung": 1, 
    "left lower lung": 2, 
    "right upper lung": 3, 
    "right middle lung": 4,  
    "right lower lung": 5,
    "mediastinum": 6, 
    "aorta": 7,
    "spine": 8,  
    "heart": 9, 
}

anatomies_radgraph = [
    'trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
    'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
    'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
    'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
    'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
    'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
    'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
    'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'
]

anatomies_radgraph_ignored = [
    "unspecified", "other"
]

diseases_radgraph = [
    'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
    'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
    'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
    'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
    'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
    'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
    'tail_abnorm_obs', 'excluded_obs'
]

diseases_radgraph_ignored = [
    'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
    'process', 'enlarge', 'tip', 'low', 'line', 'tortuous', 'lead', 'device', 'expand',
    'wire', 'degenerative', 'thicken', 'marking', 'hyperinflate', 'blunt', 'loss', 'widen',
    'aerate', 'crowd', 'obscure', 'shift', 'pressure', 'finding', 'borderline', 'hardware'
]


def main():
    filepath_anatomies_mapping = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/mapping_anatomies.csv"
    with open(filepath_anatomies_mapping) as file:
        data_anatomies_mapping = file.readlines()
    data_anatomies_mapping = [line.strip().split(",") for line in data_anatomies_mapping]
    data_anatomies_mapping = {anatomy_mimic: anatomy_target for anatomy_mimic, anatomy_target in data_anatomies_mapping}

    filepath_mimic = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/src/data_pipeline_hub/COMG_Anatomy_pipeline/MIMIC_meta_data.json"
    with open(filepath_mimic) as file:
        data_mimic = json.load(file)
    data_mimic_split = data_mimic["train"] + data_mimic["valid"]
    data_mimic_dict = {}
    for sample in tqdm.tqdm(data_mimic_split):
        dicom_id = sample["dicom_id"]
        data_mimic_dict[dicom_id] = sample

    filepath_radgraph = "/mnt/12T/01_hieu/VLM/MedKLIP/Pretrain/data_file/MedKLIP/rad_graph_metric_train.json"
    with open(filepath_radgraph) as file:
        data_train_radgraph = json.load(file)
    data_train_radgraph_dict = {}
    num_ignored = 0
    for k, v in data_train_radgraph.items():
        dicom_id = os.path.basename(k).split(".")[0]
        if dicom_id not in data_mimic_dict.keys():
            num_ignored += 1
            continue
        data_train_radgraph_dict[dicom_id] = v
    print(f"num_ignored: {num_ignored} / {len(data_train_radgraph)}")

    filepath_radgraph_matrix = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/MAVL/landmark_observation_adj_mtx_nozip.npy"
    data_radgraph_outputs = np.load(filepath_radgraph_matrix)

    data_radgraph_outputs_dict = {}
    for dicom_id, sample in data_train_radgraph_dict.items():
        labels_id = sample["labels_id"]
        sample_radgraph_output = data_radgraph_outputs[labels_id]
        data_radgraph_outputs_dict[dicom_id] = sample_radgraph_output

    data_mimic_mapping = {}  # map disease and anatomy
    for dicom_id, radgraph_matrix in data_radgraph_outputs_dict.items():
        sample_data = []
        anatomies, diseases = np.where(radgraph_matrix == 1)
        for anatomy_index, disease_index in zip(anatomies, diseases):
            disease_name = diseases_radgraph[disease_index]
            anatomy_name_mimic = anatomies_radgraph[anatomy_index]
            if disease_name in diseases_radgraph_ignored:
                continue
            if anatomy_name_mimic in anatomies_radgraph_ignored:
                continue
            anatomy_name_target = data_anatomies_mapping[anatomy_name_mimic]
            sample_data.append(
                {
                    "anatomy": anatomy_name_target.replace("_", " "),
                    "disease": disease_name.replace("_", " ")
                }
            )
        data_mimic_mapping[dicom_id] = sample_data
    
    filepath_output = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/mimic_disease_anatomy_pairs.json"
    with open(filepath_output, "w") as file:
        json.dump(data_mimic_mapping, file, indent=4)
    

if __name__ == "__main__":
    main()
