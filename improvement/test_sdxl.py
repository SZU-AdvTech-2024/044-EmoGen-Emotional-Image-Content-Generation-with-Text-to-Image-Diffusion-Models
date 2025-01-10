import os
import json
import random
import torch
import pickle
import argparse
from diffusers import DiffusionPipeline, StableDiffusionPipeline

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

with open("/mnt/d/EmoGen/data_process/emoset_caption.json", "r") as f:
    emoset_caption = json.load(f)

caption_list = {}
for emo in emotion_list:
    caption_list[emo] = []

for filename, caption in emoset_caption.items():
    emo = filename.split("_")[0]
    caption_list[emo].append(caption)

# for emotion in emotion_list:

pipe = DiffusionPipeline.from_pretrained(
    "/mnt/d/models/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.enable_model_cpu_offload()

save_root = f"/mnt/d/EmoGen/improvement/results/generate_sdxl"
os.makedirs(save_root, exist_ok=True)

save_dir = f"{save_root}/{emotion}"
os.makedirs(save_dir, exist_ok=True)

for i in range(1000):
    caption_id = random.randint(0, len(caption_list[emotion]) - 1)
    caption = caption_list[emotion][caption_id]
    caption = caption.replace("/", "")

    image = pipe(f"{emotion}, {caption}").images[0]
    unique_random_string = generate_unique_random_string()
    image.save(f"{save_dir}/{caption}_{unique_random_string}.png")
