IMAGE_NAME="med-cxr-lmm"
TAG="latest"
CONTAINER_NAME=${IMAGE_NAME}-${TAG}

docker run \
    --name ${CONTAINER_NAME} \
    --detach \
    --interactive \
    --tty \
    --gpus all \
    --shm-size=8g \
    -v /mnt/12T/02_duong/data-center:/home/data-center \
    -v /mnt/12T/02_duong/Medical-ChestXray-LMM/opensources/Chest-Xray-LISA/dataset/VinDr:/mnt/12T/02_duong/Medical-ChestXray-LMM/opensources/Chest-Xray-LISA/dataset/VinDr \
    -v /mnt/12T/02_duong/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/checkpoints:/home/checkpoints \
    -v $(pwd)/model_registry:/home/model_registry \
    -v $(pwd):/Medical-ChestXray-LMM \
    -e HF_HOME=/root/.cache/huggingface \
    ${IMAGE_NAME}:${TAG} \
    /bin/bash
