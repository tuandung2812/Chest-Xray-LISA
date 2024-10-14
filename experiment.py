import subprocess
import os

# Define the arguments as a list
args = [
    "python", "LISAMed/train_ds.py",  # Python executable and script
            "--version=../../data/pretrained_weights/llava-med-v1.5-mistral-7b",
            # "--version=../../data/pretrained_weights/LLaVA-Lightning-7B-delta-v1-1/",
            "--dataset_dir=../../data/brats2023/2d/",
            "--vision_pretrained=../../data/pretrained_weights/sam_vit_h_4b8939.pth",
            "--dataset=reason_seg",
            "--sample_rates=1",
            "--exp_name=lisa-oneclass",
            "--batch_size=6",
            "--steps_per_epoch", "500",
            "--conv_type", "llava_v1",
]

# Set CUDA_VISIBLE_DEVICES in the environment
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

# Run the script using subprocess
try:
    result = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,env=env)

    # Print the output and errors (if any)
    print("Output:\n", result.stdout)
    if result.stderr:
        print("Errors:\n", result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running the script: {e}")
    print("Stdout:\n", e.stdout)
    print("Stderr:\n", e.stderr)