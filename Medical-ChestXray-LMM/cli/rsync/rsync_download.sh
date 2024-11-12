#!/bin/bash

USERNAME=duong
IP_DEST="10.18.224.65"
DIRPATH_FROM=/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_MedGLaMM_caption_v3.zip
DIRPATH_DEST=tmp

rsync -avz --progress ${USERNAME}@${IP_DEST}:${DIRPATH_FROM} ${DIRPATH_DEST}

# scp -r ${user_name}@${ip_dest}:${from_path} ${dest_path}