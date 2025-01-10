import os
import json
import random
import torch
import pickle
import argparse
from diffusers import StableDiffusionPipeline

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

import uuid


def generate_unique_random_string():
    return uuid.uuid4().hex


parser = argparse.ArgumentParser(description="emotion.")
parser.add_argument(
    "--emotion",
    type=str,
    default=None,
)
args = parser.parse_args()

assert args.emotion is not None
emotion = args.emotion
emotion_list = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]


# def read_attr():
#     properties = ["object", "scene"]
#     attribute_pro = {"object": [], "scene": []}
#     attribute_total = []
#     attribute_emo = {}
#     for property in properties:
#         with open(f"/mnt/d/EmoGen/dataset_balance/{property}_attr.pkl", "rb") as f:
#             useful_attr = pickle.load(f)
#             tmp = []
#             for key in useful_attr:
#                 tmp.extend(useful_attr[key])
#                 try:
#                     attribute_emo[key].extend(useful_attr[key])
#                 except:
#                     attribute_emo[key] = []
#                     attribute_emo[key].extend(useful_attr[key])
#             attribute_pro[property].extend(tmp)
#             attribute_total.extend(tmp)
#     return attribute_total, attribute_emo


# total_attr, _ = read_attr()

with open("/mnt/d/EmoGen/data_process/emoset_caption.json", "r") as f:
    emoset_caption = json.load(f)

caption_list = {}
for emo in emotion_list:
    caption_list[emo] = []

for filename, caption in emoset_caption.items():
    emo = filename.split("_")[0]
    caption_list[emo].append(caption)

step_list = [400, 800, 1200, 1600, 2000, 2400]

for step in step_list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = CLIPTextModel.from_pretrained(
        f"/mnt/d/EmoGen/improvement/results/emoclip-1219/checkpoint-{str(step)}",
        torch_dtype=torch.float16,
    )
    text_encoder.to(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        "/mnt/d/models/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    pipe.text_encoder = text_encoder

    save_root = (
        f"/mnt/d/EmoGen/improvement/results/emoclip-1219/edited_20241226/{str(step)}"
    )
    os.makedirs(save_root, exist_ok=True)

    co_save_root = (
        f"/mnt/d/EmoGen/improvement/results/emoclip-1219/edited_20241226_co/{str(step)}"
    )
    os.makedirs(co_save_root, exist_ok=True)

    save_dir = f"{save_root}/{emotion}"
    os.makedirs(save_dir, exist_ok=True)

    co_save_dir = f"{co_save_root}/{emotion}"
    os.makedirs(co_save_dir, exist_ok=True)
    for i in range(100):
        caption_id = random.randint(0, len(caption_list[emotion]) - 1)
        caption = caption_list[emotion][caption_id]

        image = pipe(caption + emotion).images[0]
        unique_random_string = generate_unique_random_string()
        image.save(f"{save_dir}/{emotion}_{caption}_{unique_random_string}.png")

        image = pipe(caption).images[0]
        unique_random_string = generate_unique_random_string()
        image.save(f"{co_save_dir}/{emotion}_{caption}_{unique_random_string}.png")
