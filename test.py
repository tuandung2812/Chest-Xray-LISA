import argparse
import os
import sys
from functools import partial
import cv2
import numpy as np
import shutil
from model.llava.mm_utils import tokenizer_image_token

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from utils.vindr_dataset import VinDrDataset
from utils.dataset import collate_fn
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="LISAMed/test_results", type=str)
    parser.add_argument("--precision",default="bf16",type=str,choices=["fp32", "bf16", "fp16"],help="precision for inference")
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=250, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type",default="llava_v1",type=str,choices=["llava_v1", "llava_llama_2"],)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--experiment_name", default="simple_test", type=str)
    return parser.parse_args(args)

def compute_dice_score(mask, pred):
    """Compute the Dice score between two binary segmentation masks."""
    mask = np.asarray(mask).astype(bool)
    pred = np.asarray(pred).astype(bool)
    
    if mask.shape != pred.shape:
        raise ValueError("Shape mismatch: segmentation masks must have the same shape.")
    
    intersection = np.sum(mask & pred)
    volume_sum = np.sum(mask) + np.sum(pred)
    
    if volume_sum == 0:
        return 1.0  # If both segmentations are empty, return Dice score of 1.0
    
    return 2. * intersection / volume_sum

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)
    if os.path.exists(os.path.join(args.vis_save_path,args.experiment_name)):
        shutil.rmtree(os.path.join(args.vis_save_path,args.experiment_name))

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    # args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # Adding new tokens
    segmentation_tokens = ["[SEG1]", "[SEG2]", "[SEG3]"]
    phrase_tokens = ['<p>', '</p>']
    num_added_tokens = tokenizer.add_tokens(segmentation_tokens + phrase_tokens) 
    # Define the index of the [SEG] token
    args.seg_token_idx = tokenizer(segmentation_tokens, add_special_tokens=False).input_ids

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)


    if args.precision == "bf16":
        # model = model.bfloat16().cuda()
        model = model.to(device= args.local_rank).to(torch.bfloat16)

    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        model = model.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)
    conv = conversation_lib.conv_templates[args.conv_type].copy()

    # Define the images fo the test set
    test_dataset = VinDrDataset(
            base_image_dir = args.dataset_dir,
            tokenizer = tokenizer,
            vision_tower = args.vision_tower,
            mode = "Test",
            image_size = args.image_size,
            num_classes = 3,
            just_using_tumor_imgs = True,
            just_use_pos_cases = True
        )
    
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end
            ),
        )
    

    # Define the model to eval
    model.eval()
    mean_dice = 0
    count = 0
    # Disable gradient computation
    with torch.no_grad():
        # Iterate through the test loader
        for batch_idx, batch_data in enumerate(test_loader):
            # Assuming batch_data contains input data and target labels
            image_clip = batch_data['images_clip'].to(device= args.local_rank)  
            image = batch_data['images'].to(device= args.local_rank)
            prompt = batch_data["questions_list"][0]
            mask = batch_data["masks_list"][0].numpy()
            image_path = batch_data["image_paths"][0]
            original_size_list = mask.shape[1:3]
            resize_list = image.shape[2:4]

            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
                image = image.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
                image = image.half()
            else:
                image_clip = image_clip.float()
                image = image.float()
            conv.messages = []
            conv.append_message(conv.roles[0], prompt[0])
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()

            output_ids, pred_mask = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=args.model_max_length,
                tokenizer=tokenizer,
            )
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            print("text_output: ", text_output)

            # Filter values over 0
            pred_mask = pred_mask[0].detach().cpu().numpy()
            pred_mask = pred_mask > 0

            # Obtain DICE
            mean_dice += compute_dice_score(mask,pred_mask)
            print(mean_dice)
            count += 1

            # Define path to save files
            base_image_path = "/".join(image_path.split("/")[-2:])
            path_mask = os.path.join(args.vis_save_path,args.experiment_name,"mask",base_image_path)
            path_pred_mask = os.path.join(args.vis_save_path,args.experiment_name,"pred",base_image_path)
            path_pred_cap = os.path.join(args.vis_save_path,args.experiment_name,"pred_cap",base_image_path.replace("jpg","txt"))

            # Masks and predictions have three channels. We need to wrap them in one channel
            # Initialize the new array with zeros (default to background class 0)
            final_mask = np.zeros((mask.shape[-1],mask.shape[-1]), dtype=np.uint8)
            # Assign class values based on which mask has a value of 1 at each pixel
            final_mask[mask == 1] = 100
            final_pred = np.zeros((mask.shape[-1],mask.shape[-1]), dtype=np.uint8)
            final_pred[pred_mask == 1] = 100

            # Create folders if these dont exist
            if not os.path.exists("/".join(path_mask.split("/")[:-1])):
                os.makedirs("/".join(path_mask.split("/")[:-1]))
            if not os.path.exists("/".join(path_pred_mask.split("/")[:-1])):
                os.makedirs("/".join(path_pred_mask.split("/")[:-1]))
            if not os.path.exists("/".join(path_pred_cap.split("/")[:-1])):
                os.makedirs("/".join(path_pred_cap.split("/")[:-1]))

            # Save masks and preds
            cv2.imwrite(path_mask,final_mask)
            cv2.imwrite(path_pred_mask, final_pred)

            # Save the predicted caption
            with open(path_pred_cap, 'w') as file:
                file.write(text_output)

        # Get final DICE
        dice_score = mean_dice / count
        print(dice_score)


if __name__ == "__main__":
    main(sys.argv[1:])
