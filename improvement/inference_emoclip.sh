#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /mnt/d/EmoGen/improvement/inference_emoclip.py --emotion=amusement &
CUDA_VISIBLE_DEVICES=1 python /mnt/d/EmoGen/improvement/inference_emoclip.py --emotion=awe &
CUDA_VISIBLE_DEVICES=2 python /mnt/d/EmoGen/improvement/inference_emoclip.py --emotion=contentment &
CUDA_VISIBLE_DEVICES=3 python /mnt/d/EmoGen/improvement/inference_emoclip.py --emotion=excitement &
CUDA_VISIBLE_DEVICES=4 python /mnt/d/EmoGen/improvement/inference_emoclip.py --emotion=anger &
CUDA_VISIBLE_DEVICES=5 python /mnt/d/EmoGen/improvement/inference_emoclip.py --emotion=disgust &
CUDA_VISIBLE_DEVICES=6 python /mnt/d/EmoGen/improvement/inference_emoclip.py --emotion=fear &
CUDA_VISIBLE_DEVICES=7 python /mnt/d/EmoGen/improvement/inference_emoclip.py --emotion=sadness
