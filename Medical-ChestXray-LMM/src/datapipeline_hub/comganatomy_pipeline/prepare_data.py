import json

import pandas as pd


def load_data(path_data="MIMIC_meta_data.json"):
    with open(path_data) as file:
        data = json.load(file)
    return data


def create_meta_data():
    master_path = "../datasets/MIMIC_CXR/master.csv"
    data_master_df = pd.read_csv(master_path)
    print(data_master_df.columns)

    data_meta = {
        "train": [],
        "valid": []
    }

    for _, row in data_master_df.iterrows():
        data_meta[row["split"]].append(row.to_dict())
    
    path_out = "MIMIC_meta_data.json"
    with open(path_out, 'w') as f:
        json.dump(data_meta, f, indent=4, sort_keys=True)


def main():
    load_data()


if __name__ == "__main__":
    main()
