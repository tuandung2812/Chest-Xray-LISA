import json


def load_probmed_data(
    filepath_metadata
):
    with open(filepath_metadata, "r") as f:
        data = json.load(f)
        f.close()
    return data