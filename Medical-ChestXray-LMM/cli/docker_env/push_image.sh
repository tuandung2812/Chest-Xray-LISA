LOCAL_IMAGE_NAME=med-cxr-lmm-dgx
TAG=latest
HUB_IMAGE_NAME=antimachinee/${LOCAL_IMAGE_NAME}

docker tag ${LOCAL_IMAGE_NAME}:${TAG} ${HUB_IMAGE_NAME}:${TAG}
docker push ${HUB_IMAGE_NAME}:${TAG}
