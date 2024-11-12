import sys
sys.path.append(".")

import argparse
import torch
import os
import json
from tqdm import tqdm

from evaluation_hub.load_data.load_vindr_data import load_vindr_data
from model_hub.llavamed.llavamed_0.conversation import conv_templates, SeparatorStyle
from model_hub.llavamed.llavamed_0.model.builder import load_pretrained_model
from model_hub.llavamed.llavamed_0.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from model_hub.llavamed.llavamed_0.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import transformers

from PIL import Image

transformers.logging.set_verbosity_error()


def eval_model(
        args,
        filepath_metadata,
        dirpath_images,
        dirpath_output,
    ):
    os.makedirs(dirpath_output, exist_ok=True)
    filepath_output = os.path.join(dirpath_output, f"{os.path.basename(filepath_metadata).split('.')[0]}_llavamed0.json")
    metadata = load_vindr_data(
        filepath_metadata=filepath_metadata
    )

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len, image_token_len = load_pretrained_model(model_path, args.model_base, model_name)

    data_output = {}
    for filename_image, questions in tqdm(metadata.items()):
        try:
            data_output[filename_image] = []
            for sample in questions:
                filepath_image = os.path.join(dirpath_images, filename_image + ".png")
                question = sample["question"]
                question = question.replace('<image>', '').strip()
                
                qs = question
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                image = Image.open(filepath_image).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config)[0]

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=True,
                        temperature=args.temperature,
                        max_new_tokens=1024)

                outputs = tokenizer.batch_decode(
                    output_ids, 
                    skip_special_tokens=True)[0].strip()

                # print(outputs)

                sample["response"] = outputs
                data_output[filename_image].append(sample)
                
                with open(filepath_output, "w") as f:
                    json.dump(data_output, f) 
        except Exception as e:
            print(f"Error: {e}. File: {filename_image}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    eval_model(
        args,
        filepath_metadata="/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/vqa_multichoices_vindr_unhealthy_image_normal.json",
        dirpath_images="/mnt/12T/02_duong/data-center/VinDr/train_png_16bit",
        dirpath_output="/mnt/12T/02_duong/data-center/Medical-ChestXray-Dataset-for-LMM-Data/evaluation_hub/probmed_v3"
    )
