secret_name=hieu-tms-docker-token

kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=/media/hieu/6ac7d369-b609-4b09-97b0-27ed881b25f9/duong/TumorSegmentationML/cli/docker-env/config.json \
    --type=kubernetes.io/dockerconfigjson

kubectl create secret generic hieu-tms-wandb-api-key --from-literal=token=c5ae15453319c6500f8b46f9605a61b221ffe421