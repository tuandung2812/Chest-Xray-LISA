#!/bin/bash

USERNAME=duong
IP_DEST="10.12.65.152"
DIRPATH_FROM=/mnt/12T/02_duong/data-center/VinDr/train_png_16bit
DIRPATH_DEST=/media/hieu/6ac7d369-b609-4b09-97b0-27ed881b25f9/duong/data-center/VinDr

rsync -avz --progress ${DIRPATH_FROM} ${USERNAME}@${IP_DEST}:${DIRPATH_DEST}
