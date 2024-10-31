python test.py   --version "runs/regularize_medsam_vindr_llavamed_bs_16_lr_5e-5" \
  --dataset_dir "./dataset" \
  --dataset "reason_seg" \
    --experiment_name "regularize_medsam_vindr_llavamed_bs_16_lr_5e-5"

python merge_lora_weights.py --