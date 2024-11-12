IMAGE_NAME=med-cxr-lmm
TAG=latest
CONTAINER_NAME=${IMAGE_NAME}-${TAG}

docker exec \
    -it \
    ${CONTAINER_NAME} \
    bash
