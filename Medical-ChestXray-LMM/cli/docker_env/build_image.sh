IMAGE_NAME="med-cxr-lmm"
TAG="latest"
DOCKERFILE_PATH=".docker/Dockerfile"

docker build \
    -f ${DOCKERFILE_PATH} \
    --tag ${IMAGE_NAME}:${TAG} \
    .
