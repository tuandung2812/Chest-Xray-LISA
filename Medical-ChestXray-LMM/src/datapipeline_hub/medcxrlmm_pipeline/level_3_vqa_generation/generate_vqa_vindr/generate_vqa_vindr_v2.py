import sys
sys.path.append(".")

import pandas as pd

from dataset_hub.vindr.config_vindr.config_vindr_medcxrlmm import *
from datapipeline_hub.medcxrlmm_pipeline.level_3_vqa_generation.question_answer_templates.qa_template_vindr_medcxrlmm import *

RAD_ID = "R9"


def generate_vqa_no_findings():
    # close question
    
    # open question
    # close question adversarial
    # open question adversarial
    return


def generate_vqa_have_findings(disease_name, anatomy_name):
    # close question
    # open question
    # close question adversarial
    # open question adversarial
    return


def main():
    df0 = pd.read_csv(FILEPATH_VINDR_MEDCXRLMM_CAPTION_MEDATADATA)
    df_filtered_no_findings = df0[df0['rad_id'] != RAD_ID]
    images_no_findings = df_filtered_no_findings['filepath_image'].unique()
    print(len(images_no_findings))

    df_filtered_have_findings = df0[df0['rad_id'] == RAD_ID]
    print(len(df_filtered_have_findings))

    vqa_records = []
    for _ in images_no_findings:
        vqa_pairs_no_findings = generate_vqa_no_findings()
    vqa_records.extend(vqa_pairs_no_findings)

    for index, df_diagnosis in df_filtered_have_findings.iterrows():
        disease_name = df_diagnosis['disease_name']
        anatomy_name = df_diagnosis['anatomy_name']
        assert disease_name is not None
        assert anatomy_name is not None
        vqa_pairs_have_findings = generate_vqa_have_findings(disease_name, anatomy_name)
    vqa_records.extend(vqa_pairs_have_findings)


if __name__ == "__main__":
    main()
