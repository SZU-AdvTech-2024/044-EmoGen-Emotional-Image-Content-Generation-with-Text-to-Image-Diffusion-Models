import os
import csv
import torch
from config import Config
from classifier import Classifier
from transformers import CLIPModel, CLIPProcessor
from PIL import ImageFile
import pandas as pd
from tqdm import tqdm
from img_utils import load_img

ImageFile.LOAD_TRUNCATED_IMAGES = True
weight = "./model_19.pth"
device = "cuda"

data_root = r"/mnt/d/data/EmoSet_v5_train-test-val/image"

image_list = []
for root, _, file_path in os.walk(data_root):
    for file in file_path:
        emotion = file.split("_")[0]
        if file.endswith(".jpg"):
            file = file.split(".jpg")[0]
            image_list.append(file)

cfg = Config("./project_config.yaml")
# clip_path = cfg.clip_path
clip_path = r"/mnt/d/models/clip-vit-large-patch14"
classifier = Classifier(768, 8).to(device)
state = torch.load(weight, map_location=device)
# classifier.load_state_dict(state)
classifier.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
classifier.eval()

# CLIPmodel = CLIPModel.from_pretrained(clip_path).to(device)
# processor = CLIPProcessor.from_pretrained(clip_path)

image_features = {}
feature_root = "/mnt/d/EmoGen/data_process/features"
feature_dir = os.listdir(feature_root)
for feature_path in feature_dir:
    emo = feature_path.split("_")[0]
    image_features[emo] = torch.load(os.path.join(feature_root, feature_path))


csv_file = "./evaluation_emoset.csv"
resume = False
mode = "w"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    resume = True
    mode = "a"
else:
    df = None
with open(csv_file, mode=mode, newline="") as file:
    writer = csv.writer(file)
    if not resume:
        writer.writerow(["Filename", "Emotion", "Emotion_score"])
    for filename in tqdm(image_list, total=len(image_list)):
        if df is not None and filename in df["Filename"].values:
            print(filename, "has been evaluated")
            continue
        try:
            # image_path = os.path.join(data_root, filename)
            # image = load_img(image_path)
            # image = processor(images=image, return_tensors="pt", padding=True).to(
            #     device
            # )
            # clip = CLIPmodel.get_image_features(**image)
            emotion = filename.split("_")[0]
            clip = image_features[emotion][filename]
            pred = classifier(clip.to(device))
            pred = torch.softmax(pred, dim=1)
            pred_raw = pred.squeeze(0).cpu().detach().numpy().tolist()
            # print(pred)
            pred_emotion_8 = torch.argmax(pred, dim=1, keepdim=True).item()
            emotion_score = pred_raw[pred_emotion_8]
            # print(pred_emotion_8)
            writer.writerow([filename, cfg.emotion_list[pred_emotion_8], emotion_score])
            print(filename, cfg.emotion_list[pred_emotion_8], emotion_score)
        except:
            print(filename, "error!!!")
        # break
