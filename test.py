from training.model import *
import torch
import torch.nn as nn

class image_encoder(nn.Module):
    def __init__(self):
        super(image_encoder, self).__init__()
        self.resnet = BackBone()
        state = torch.load("weights/image_encoder/2023-08-22-best.pth")
        self.resnet.load_state_dict(state)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[1:-1])

    def forward(self, x):
        out = self.resnet(x)
        return out

x = image_encoder()
print('OK')
