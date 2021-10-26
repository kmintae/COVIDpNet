"""COVIDxCXR.py

Loading & pre-processing COVIDx-CXR datasets.

AUTHOR:
    Mintae Kim
"""
import yaml

import torchvision
import torch.utils.data
from torchvision.transforms import transforms as T

with open("config/loader.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['COVIDx-CXR']

class CXRDataset(torch.utils.data.Dataset):
    def __init__(self): # Pre-processing
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

def load():
    path = config['path']

    # Dataset Preprocessing
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = torchvision.datasets.ImageNet(".", split="train", transform=transform)