import sys
import torch
import lpips
import os
import numpy as np
import torch.nn as nn
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image, ImageOps
from tqdm.auto import tqdm
import argparse
import random
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import torch.nn.functional as F
import torchvision.models as models


class emo_classifier(nn.Module):
    def __init__(
        self,
    ):
        super(emo_classifier, self).__init__()
        self.fc = nn.Linear(768, 8)

    def forward(self, x):
        x = self.fc(x)
        return x


def crop_img(img, output_size=(512, 512)):
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize(output_size)
    return img_resized


def load_img(image_path):
    image = Image.open(image_path)
    image = crop_img(image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


@torch.no_grad()
def emo_cls(cur_dir, device, weight):
    classifier = emo_classifier().to(device)
    state = torch.load(weight, map_location=device)
    classifier.load_state_dict(state)
    classifier.eval()

    CLIPmodel = CLIPModel.from_pretrained("/mnt/d/models/clip-vit-large-patch14").to(
        device
    )
    processor = CLIPProcessor.from_pretrained("/mnt/d/models/clip-vit-large-patch14")

    class EmoDataset(Dataset):
        def __init__(self, data_root, processor):
            self.emotion_list_8 = {
                "amusement": 0,
                "awe": 1,
                "contentment": 2,
                "excitement": 3,
                "anger": 4,
                "disgust": 5,
                "fear": 6,
                "sadness": 7,
            }
            self.emotion_list_2 = {
                "amusement": 0,
                "awe": 0,
                "contentment": 0,
                "excitement": 0,
                "anger": 1,
                "disgust": 1,
                "fear": 1,
                "sadness": 1,
            }
            self.image_paths = []
            self.processor = processor
            self.data_root = data_root
            for root, _, file_path in os.walk(self.data_root):
                for file in file_path:
                    if file.endswith("jpg") or file.endswith("png"):
                        self.image_paths.append(os.path.join(root, file))
            self._length = len(self.image_paths)

        def __len__(self):
            return self._length

        def __getitem__(self, i):
            path = self.image_paths[i]
            example = {}
            # image = Image.open(path).convert("RGB")
            image = load_img(path)
            data = self.processor(images=image, return_tensors="pt", padding=True)
            data["pixel_values"] = data["pixel_values"].squeeze(0)
            example["image"] = data
            # data = self.model.get_image_features(**data)
            example["emotion_8"] = self.emotion_list_8[path.split("/")[-2]]
            example["emotion_2"] = self.emotion_list_2[path.split("/")[-2]]
            return example

    val_dataset = EmoDataset(cur_dir, processor)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, pin_memory=True
    )
    picture_num = len(val_dataset)
    val_loader = tqdm(val_loader, file=sys.stdout)
    score_8 = 0
    score_2 = 0
    acc_num_2 = 0
    acc_num_8 = 0

    def eightemotion(Emo, Emo_num, Emo_score, pre, label, correct):
        for i in range(label.shape[0]):
            emo_label = label[i][0].item()
            Emo[emo_label] += correct[i].item()
            Emo_num[emo_label] += 1
            Emo_score[emo_label] += pre[i][emo_label]
        return Emo, Emo_num, Emo_score

    Emo = [0] * 8
    Emo_num = [0] * 8
    Emo_score = [0] * 8
    Emotion = [
        "amusement",
        "awe",
        "contentment",
        "excitement",
        "anger",
        "disgust",
        "fear",
        "sadness",
    ]
    for step, data in enumerate(val_loader):
        images = data["image"].to(device)
        clip = CLIPmodel.get_image_features(**images)
        pred = classifier(clip.to(device))
        labels_8 = data["emotion_8"].to(device).unsqueeze(1)
        labels_2 = data["emotion_2"].to(device).unsqueeze(1)
        pred_emotion_8 = torch.argmax(pred, dim=1, keepdim=True)
        p_8 = F.softmax(pred)
        p_2 = p_8.reshape((p_8.shape[0], 2, 4))
        p_2 = torch.sum(p_2, dim=2)
        p_2 = p_2.reshape((p_8.shape[0], -1))

        pred_emotion_2 = torch.argmax(p_2, dim=1, keepdim=True)

        pred_score_8 = torch.gather(p_8, dim=1, index=labels_8)
        pred_score_2 = torch.gather(p_2, dim=1, index=labels_2)

        acc_num_2 += (labels_2 == pred_emotion_2).sum().item()
        score_2 += torch.sum(pred_score_2).item()
        acc_num_8 += (labels_8 == pred_emotion_8).sum().item()
        score_8 += torch.sum(pred_score_8).item()
        eightemotion(
            Emo, Emo_num, Emo_score, p_8, labels_8, (labels_8 == pred_emotion_8)
        )
    acc_8 = (acc_num_8 / picture_num) * 100
    total_score_8 = score_8 / picture_num
    acc_2 = (acc_num_2 / picture_num) * 100
    total_score_2 = score_2 / picture_num
    with open(os.path.join(cur_dir, "evaluation.txt"), "a") as f:
        f.write(f"emo_score (8 class): {total_score_8:.2f}" + "\n")
        f.write(f"accuracy (8 class): {acc_8:.2f}%" + "\n")
        f.write(f"emo_score (2 class): {total_score_2:.2f}" + "\n")
        f.write(f"accuracy (2 class): {acc_2:.2f}%" + "\n")
        for i in range(8):
            tmp = Emo[i] / Emo_num[i] * 100
            f.write(
                f"{Emotion[i]} accuracy:{tmp:.2f}% score:{(Emo_score[i]/Emo_num[i]):.2f} \n"
            )


class EmoDataset(Dataset):
    def __init__(self, data_root):
        self.image_paths = []
        self.data_root = data_root
        for root, _, file_path in os.walk(self.data_root):
            for file in file_path:
                if file.endswith("jpg") or file.endswith("png"):
                    self.image_paths.append(os.path.join(root, file))
        self._length = len(self.image_paths)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        path = self.image_paths[i]
        example = {}
        # image = Image.open(path).convert("RGB")
        # example["image"] = self.tfm(image)
        image = load_img(path)
        example["image"] = (
            torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        )
        return example


def get_biggest_prob(x):
    max_values, _ = torch.max(x, dim=1)
    max_values = max_values.unsqueeze(1)
    return max_values


def Lpips(img0, img1, loss_fn_alex):  # pair_image should be two image, RBG, range(0~1)
    d = loss_fn_alex(img0, img1)
    return d.item()


def DistanceOfCos(
    img0, img1, model, processor
):  # pair_image should be two image, RBG, range(0~1)
    data_pro = processor(images=[img0, img1], return_tensors="pt", padding=True).to(
        model.device
    )
    data_pro = model.get_image_features(**data_pro)
    d = 1 - F.cosine_similarity(
        data_pro[0, :].unsqueeze(0), data_pro[1, :].unsqueeze(0)
    )
    mse = F.mse_loss(data_pro[0, :].unsqueeze(0), data_pro[1, :].unsqueeze(0))
    return d.item(), mse.item()


def Semantic_diversity(curdir, num_sample, device):
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
    images_path = []
    # curdir = os.path.join(wkdir, subdir)
    model = CLIPModel.from_pretrained("/mnt/d/models/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("/mnt/d/models/clip-vit-large-patch14")
    loss_fn_alex = lpips.LPIPS(net="alex")
    record = {"lpips_score": [], "difference_score": [], "mse_score": []}
    # search one emotion's semantic_diversity
    for emotion in emotion_list:
        # get all image in the dir
        cur_path = os.path.join(curdir, emotion)
        for root, _, file_path in os.walk(cur_path):
            for file in file_path:
                if file.endswith("jpg") or file.endswith("png"):
                    path = os.path.join(root, file)
                    images_path.append(path)

        # randomly select two picture and do distance calculation
        total_lpips = []
        difference_list = []
        mse_list = []
        for _ in range(num_sample):
            random_image_pair = random.sample(images_path, 2)
            tfm = transforms.ToTensor()
            img0 = tfm(Image.open(random_image_pair[0]))
            img1 = tfm(Image.open(random_image_pair[1]))
            tmp_lpips_score = Lpips(img0, img1, loss_fn_alex)
            tmp_difference_score, mse_distance = DistanceOfCos(
                img0, img1, model, processor
            )
            total_lpips.append(tmp_lpips_score)
            difference_list.append(tmp_difference_score)
            mse_list.append(mse_distance)
        lpips_score = sum(total_lpips) / num_sample
        mse_score = sum(mse_list) / num_sample
        difference_score = sum(difference_list) / num_sample
        with open(f"{curdir}/evaluation.txt", "a") as f:
            f.write(f"---------{emotion}------------- \n")
            f.write(f"LPIPS score: {lpips_score:.3f} \n")
            f.write(f"Semantic diversity score (cos): {difference_score:.4f} \n")
            f.write(f"Semantic diversity score (MSE): {mse_score:.4f} \n")
        record["mse_score"].append(mse_score)
        record["lpips_score"].append(lpips_score)
        record["difference_score"].append(difference_score)
    lpips_score = sum(record["lpips_score"]) / len(record["lpips_score"])
    difference_score = sum(record["difference_score"]) / len(record["difference_score"])
    mse_score = sum(record["mse_score"]) / len(record["mse_score"])
    print(f"LPIPS score: {lpips_score:.3f} \n")
    print(f"Semantic diversity score (MSE): {mse_score:.4f} \n")
    with open(f"{curdir}/evaluation.txt", "a") as f:
        # 在文件末尾追加写入文本内容
        f.write(f"---------Average------------- \n")
        f.write(f"LPIPS score: {lpips_score:.3f} \n")
        f.write(f"Semantic diversity score (cos): {difference_score:.4f} \n")
        f.write(f"Semantic diversity score (MSE): {mse_score:.4f} \n")


@torch.no_grad()
def Semantic_clarity(cur_dir, device):
    # cur_dir = os.path.join(wkdir, subdir)
    val_dataset = EmoDataset(cur_dir)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, pin_memory=True
    )
    # picture_num = len(val_dataset)
    print("val dataset length: ", len(val_dataset))
    val_loader = tqdm(val_loader, file=sys.stdout)

    # 1. scene classifier
    arch = "resnet50"

    # load the pre-trained weights
    model_file = "%s_places365.pth.tar" % arch
    if not os.access(model_file, os.W_OK):
        weight_url = "http://places2.csail.mit.edu/models_places365/" + model_file
        os.system("wget " + weight_url)
    scene_classifier = models.__dict__[arch](num_classes=365).to(device)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    scene_classifier.load_state_dict(state_dict)
    scene_classifier.eval()

    # 2.object classifier
    object_classifier = models.resnet50(pretrained=True).to(device)
    object_classifier.eval()

    pred_list = []
    for step, data in enumerate(val_loader):
        pred_scene = scene_classifier(data["image"].to(device))
        prob_pred_scene = F.softmax(pred_scene)
        pred_object = object_classifier(data["image"].to(device))
        prob_pred_object = F.softmax(pred_object)
        big_scene = get_biggest_prob(prob_pred_scene)
        big_object = get_biggest_prob(prob_pred_object)
        for i in range(data["image"].shape[0]):
            pred_list.append(
                big_scene[i].cpu().item()
                if big_scene[i] > big_object[i]
                else big_object[i].cpu().item()
            )
    if len(pred_list) != 0:
        clarity_score = sum(pred_list) / len(pred_list)
    else:
        clarity_score = 0
    print(f"Semantic_Clarity_score: {clarity_score:.3f} \n")
    with open(f"{cur_dir}/evaluation.txt", "a") as f:
        # 在文件末尾追加写入文本内容
        f.write(f"Semantic_Clarity_score: {clarity_score:.3f} \n")


weight = "/mnt/d/EmoGen/weights/Clip_emotion_classifier/time_2023-11-12_03-29-best.pth"


# result_root = "/mnt/d/EmoGen/improvement/results"
# epochs = [400, 800, 1200, 1600, 2000, 2400]
result_root = "/mnt/d/EmoGen/runs"

result_dir = [
    # "emoclip-1219/edited",
    # "emoclip-1219/edited_20241226",
    # "emoclip-1219/edited_20241226_co",
    # "generate_sdxl",
    "test1/1/img",
    "test1_emoclip2500/1/img",
    "test2/1/img",
    "test2_emoclip2500/1/img",
]
device = "cuda"
num_sample = 10

# for epoch in epochs:
#     for i in range(len(result_dir)):
#         output_dir = os.path.join(result_root, result_dir[i], str(epoch))
#         emo_cls(output_dir, device, weight)
#         Semantic_clarity(output_dir, device)
#         Semantic_diversity(output_dir, num_sample, device)

for i in range(len(result_dir)):
    output_dir = os.path.join(result_root, result_dir[i])
    emo_cls(output_dir, device, weight)
    Semantic_clarity(output_dir, device)
    Semantic_diversity(output_dir, num_sample, device)
