#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

accelerate launch --config_file /mnt/d/EmoGen/improvement/run_config.yaml /mnt/d/EmoGen/improvement/ft_emoclip.py \
    --pretrained_model_name_or_path=/mnt/d/models/clip-vit-large-patch14 \
    --data_root=/mnt/d/data/EmoSet_v5_train-test-val \
    --output_dir=/mnt/d/EmoGen/improvement/results/emoclip-1219 \
    --resolution=768 \
    --train_batch_size=128 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=16 \
    --checkpointing_steps=400 --checkpoints_total_limit=8 \
    --learning_rate=5e-05 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=2024

# 5e-05
# 1e-04
# --max_train_steps=15000 \
