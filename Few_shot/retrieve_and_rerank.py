"""
初始化设置
"""
import os
import io
import re
import ast
import pdb
import clip
import json
import torch
import faiss
import chromadb
import numpy as np
from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel

### 初始化一个clip模型
clip_model, preprocess = clip.load("ViT-B/16", device="cuda")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import requests
from io import BytesIO


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(query, image_files, model, model_name, conv_mode=None, sep = ",", temperature=0, top_p=None, num_beams=1, max_new_tokens=512):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


disable_torch_init()

model_path = "/mnt/petrelfs/liuziyu/LLM_Memory/LLaVA/checkpoints/llava-v1.5-7b-fgvc-lora"
model_base = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)

# model_path = "liuhaotian/llava-v1.5-7b"
# model_base = None
# model_name = get_model_name_from_path(model_path)

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path, model_base, model_name
# )

shot_number = 4
top_k = 5
dataset_name = 'caltech101'


classnames_file_path = f"/mnt/petrelfs/liuziyu/LLM_Memory/SimplyRetrieve/CLIP-Cls/benchmarks_test/{dataset_name}_database/classnames.txt"
with open(classnames_file_path, 'r') as file:
    classnames = file.readlines()
print(len(classnames))
classnames = [classname.strip() for classname in classnames]
print(classnames)

predictions_save_path = f"/mnt/petrelfs/liuziyu/LLM_Memory/SimplyRetrieve/CLIP-Cls/output/ZeroshotCLIP_topk/vit_b16/{dataset_name}/predictions_{shot_number}_shot_knn_rerank.pth"
predictions_knn_path = f"/mnt/petrelfs/liuziyu/LLM_Memory/SimplyRetrieve/CLIP-Cls/output/ZeroshotCLIP_topk/vit_b16/{dataset_name}/predictions_{shot_number}_shot_knn.pth"
predictions = torch.load(predictions_knn_path)

VLM_response_format_error = 0
for prediction in tqdm(predictions,desc="Process:"):
    ### 解析pth文件，获取图片位置和原来的预测结果
    for item in prediction.values():
        pre_class = item['pred_class']
        print(item['label'])
    for item in prediction.keys():
        test_img_path = item

    with torch.no_grad():
        names = [classnames[int(item)] for item in pre_class]
        print(pre_class)
        print(names)

        names = names[:top_k]
        
        query = f'Please play the role of a classification expert, and sort the provided {top_k} categories from high to low according to the top {top_k} similarity with the input image. Here are the optional categories:{names}.'
        image_files = [test_img_path]
        response = eval_model(query, image_files, model, model_name)
        print("\033[92m" + response + "\033[0m")

        try:
            match = re.search(r"\[(.*?)\]", response)
            if match:
                content_inside_brackets = match.group(1)
                new_names = []
                for item in re.findall(r"'[^']+'|[^,]+", content_inside_brackets):
                    cleaned_item = item.strip("' ").strip()
                    new_names.append(cleaned_item)
                print(new_names)
                
            labels = [classnames.index(new_name) for new_name in new_names]
            labels = torch.tensor(labels)
            print(labels)
        except Exception as e:
            labels = pre_class
            VLM_response_format_error+=1
            print("error")
            print(labels)
    
        ### 修改pth文件
        for item in prediction.values():
            item['pred_class'] = labels
    
torch.save(predictions, predictions_save_path)
print("VLM_response_format_error: "+str(VLM_response_format_error))
