import os
from glob import glob

import pandas as pd


def main():
    path_root = "/mnt/16T/2024_03_04_XR_U18"

    paths_report = glob(os.path.join(path_root, "SR", "*.txt"))
    paths_pair = []
    for path_report in paths_report:
        path_image = os.path.dirname(path_report).replace("SR", "XR")


if __name__ == "__main__":
    main()
