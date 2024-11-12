import os
import glob

import pandas as pd


def main():
    dirpath_predict = "/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/probmed/probmed_open"
    filepaths_predict = glob.glob(f"{dirpath_predict}/*.csv")

    dict_tmp = {
        "chexagenthf": None,
        "llavahf": None,
        "llavamed0": None,
    }

    for filepath_predict in filepaths_predict:
        filename_predict = os.path.basename(filepath_predict)
        for key in dict_tmp.keys():
            if key in filename_predict:
                dict_tmp[key] = pd.read_csv(filepath_predict)

    columns = list(dict_tmp["chexagenthf"].columns[:-2]) + ["chexagenthf_pred", "chexagenthf_extracted", "llavahf_pred", "llavahf_extracted", "llavamed0_pred", "llavamed0_extracted"]
    output_df = pd.DataFrame(columns=columns)
    for col in columns[:len(dict_tmp["chexagenthf"].columns[:-2])]:
        output_df[col] = dict_tmp["chexagenthf"][col]
    output_df["chexagenthf_pred"] = dict_tmp["chexagenthf"]["response"]
    output_df["llavahf_pred"] = dict_tmp["llavahf"]["response"]
    output_df["llavamed0_pred"] = dict_tmp["llavamed0"]["response"]
    output_df["chexagenthf_extracted"] = dict_tmp["chexagenthf"]["prediction_extracted"]
    output_df["llavahf_extracted"] = dict_tmp["llavahf"]["prediction_extracted"]
    output_df["llavamed0_extracted"] = dict_tmp["llavamed0"]["prediction_extracted"]

    output_df.to_csv(f"compared.csv", index=False)
    

if __name__ == '__main__':
    main()
