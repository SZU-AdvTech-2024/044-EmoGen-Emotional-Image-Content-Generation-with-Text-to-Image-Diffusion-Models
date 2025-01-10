# 基于EmoGen的复现和改进

本项目仅为2024级研究生计算机前沿技术课程作业所用。

本项目基于 CVPR 2024 论文“EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models”原项目基础上进行修改。

修改的代码主要放在data_process和improvement两个文件夹下。

本项目不包含任何模型权重以及实验结果。

如果要对本项目进行评估，复现部分可于[原项目]([JingyuanYY/EmoGen: This is the official implementation of 2024 CVPR paper "EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models".](https://github.com/JingyuanYY/EmoGen))中下载模型权重，改进部分：

数据预处理——运行data_process文件夹下的computer_emo_score.py文件进行情感得分的计算，然后在修改原项目中数据读取部分即可。

微调CLIP文本编码器——运行improvement文件夹下的ft_emoclip.sh文件即可完成训练（需要手动修改ft_emoclip.py中的数据路径），通过improvement文件夹下的inference_emoclip.sh文件即可生成图像（需要手动修改inference_emoclip.py中的权重路径）。



# 以下是原项目的README.md文件内容：

## EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models

> [Jingyuan Yang](https://jingyuanyy.github.io/), Jiawei Feng, [Hui Huang*](https://vcc.tech/~huihuang)  
> Shenzhen University  
> Recent years have witnessed remarkable progress in image generation task, where users can create visually astonishing images with high-quality. However, exsiting text-to-image diffusion models are proficient in generating concrete concepts (dogs) but encounter challenges with more abstract ones (emotions). Several efforts have been made to modify image emotions with color and style adjustments, facing limitations in effectively conveying emotions with fixed image contents. In this work, we introduce Emotional Image Content Generation (EIGC), a new task to generate semantic-clear and emotion-faithful images given emotion categories. Specifically, we propose an emotion space and construct a mapping network to align it with powerful Contrastive Language-Image Pre-training (CLIP) space, providing a concrete interpretation of abstract emotions. Attribute loss and emotion confidence are further proposed to ensure the semantic diversity and emotion fidelity of the generated images. Our method outperforms the state-the-art text-to-image approaches both quantitatively and qualitatively, where we derive three custom metrics, i.e.,emotion accuracy, semantic clarity and semantic diversity. In addition to generation, our method can help emotion understanding and inspire emotional art design.

<a href="https://arxiv.org/abs/2401.04608"><img src="https://img.shields.io/badge/arXiv-2401.04608-b31b1b.svg" height=22.5></a>

<p align="left">
<img src="docs/teaser.png" width="1200px"/>  
<br>
Fig 1. Given an emotion category, our network produces images that exhibit unambiguous meanings (semantic-clear), reflect the intended emotion (emotion-faithful) and incorporate varied semantics (semantic-diverse).
</p>
### Description

Official implementation of our EmoGen paper.

### Pipeline

<p align="left">
<img src="docs/method-1.png" width="1200px"/>  
<br>
Fig 2. Training process of our network. Emotion representation (stage 1) learns a well-behaved emotion space and emotion content generation (stage 2) maps this space to CLIP space, aiming to generate image contents with emotion fidelity, semantic clarity and diversity.
</p>
### Setup

To create the conda environment needed to run the code, run the following command:

```
conda env create -f environment/env.yaml
conda activate EmoGen
```

Alternatively, install the requirements from `requirements.txt`

### Usage

### Preliminary

[EmoSet](https://vcc.tech/EmoSet) is needed to train in this network, as attribute label is necessary.

We need to organize the dataset according to its attributes, and the following is its layout:

```
data_root
|
├── object
|    ├── (3) cart
|    |    ├── disgust_05770.jpg
|    |    ├── ...
|    |    └── sadness_10803.jpg
|    ├── ...
|    └── (13094) plant
|
└── scene
     ├── (1) airfield
     ├── ...
     └── (2406) street
```
The number before the attribute represents the total number of images with this attribute.

### Training

To train our network, follow these steps:

First, manually modify the code related to reading EmoSet and change the file location to the location where your EmoSet is located. For example:
In training/dataset_balance.py
```
annotion_path = f'/mnt/d/dataset/EmoSet/annotation/{emotion}/{emotion}_{number}.json' # change to "{your_EmoSet_location}/annotation/{emotion}/{emotion}_{number}.json"
```

Secondly, create training dataset:
```
python training/dataset_balance.py
```

Thirdly, start to train your own network:
```
accelerate training/main.py
```

Finally, generate emotional image:
```
python training/inference.py
```
You can modify config/config.yaml to change some details.

### Emotion Creation

<p align="left">
<img src="docs/exp-5.png" width="1500px"/>  
<br>
Fig 3. Emotion creation. (a) transfers emotion representations (i.e., amusement, fear) to a series of neutral contents while (b) fuse two emotions (i.e., amusement-awe, amusement-fear) together, which may be helpful for emotional art design.
</p>
#### Emotion Transfer

To transfer emotion into object, follow these steps:
First, change training/inference.py code:
```
use_prompt = True
generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout, use_prompt)
```

Then, you can choose your object:
```
templates = [
      "{} bag", "{} cup", "{} room", "{} street",
]
```

```
python training/inference.py
```

#### Emotion Fusion

to fuse different emotion together, follow these steps:

```
python training/inference_combine_emotion.py
```
this code has similar structure as training/inference.py.

### Results

#### Comparison with other diffusion models

<p align="left">
<img src="docs/exp-1.png" width="1000px"/>  
<br>
Fig 4. Qualitative comparisions with the state-of-the-art text-to-image generation approaches and ablation studies of our method.
</p>

<div align="center">

Table 1. Comparisons with the state-of-the-art methods on emotion generation task, involving five metrics.
| Method | FID &darr; | LPIPS &uarr; | Emo-A &uarr; | Sem-C &uarr; | Sem-D &uarr; |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Stable Diffusion | 44.05 | 0.687 | 70.77% | 0.608 | 0.0199 |
| Textual Inversion | 50.51 | 0.702 | 74.87% | 0.605 | 0.0282 |
| DreamBooth | 46.89| 0.661 | 70.50% | 0.614 | 0.0178 |
| Ours     | **41.60** | **0.717** | **76.25%** | **0.633** | **0.0335** |

</div>

<div align="center">

Table 2.  User preference study. The numbers indicate the percentage of participants who prefer our results over those compared
methods, given the same emotion category as input.
| Method | Image fidelity &uarr; | Emotion faithfulness &uarr; | Semantic diversity &uarr; |
|:-------:|:-------:|:-------:|:-------:|
| Stable Diffusion | 67.86±15.08% | 73.66±11.80% | 87.88±9.64% |
| Textual Inversion | 79.91±16.92% | 72.75±16.90% | 85.66±10.51% |
| DreamBooth | 77.23±14.00% | 80.79±8.64% | 81.68±17.06% |

</div>

## Citation

If you find this work useful, please kindly cite our paper:
```
@article{yang2024emogen,
  title={EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models},
  author={Yang, Jingyuan and Feng, Jiawei and Huang, Hui},
  journal={arXiv preprint arXiv:2401.04608},
  year={2024}
}
```
