export CUDA_VISIBLE_DEVICES=1

deepspeed \
    --master_port 8887 training_hub/train_lisa/train_lisa.py \
    --filepath-config training_hub/train_lisa/config_lisa/config_lisa_vqa_vindr_medsam_llavamed_docker.yaml \
    --local_rank 0
