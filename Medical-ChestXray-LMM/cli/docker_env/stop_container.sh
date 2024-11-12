IMAGE_NAME=med-cxr-lmm
TAG=latest
CONTAINER_NAME=${IMAGE_NAME}-${TAG}

docker stop ${CONTAINER_NAME}
