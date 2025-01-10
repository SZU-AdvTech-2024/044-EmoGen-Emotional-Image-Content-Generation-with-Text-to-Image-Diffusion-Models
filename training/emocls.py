from inference import emo_cls

device = "cuda:0"
weight = "/mnt/d/EmoGen/weights/Clip_emotion_classifier/time_2023-11-12_03-29-best.pth"

output_dir = f"/mnt/d/EmoGen/runs/edited/clip"
emo_cls(output_dir, device, weight)
