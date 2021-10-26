"""ImageNet.py

Loading & pre-processing ImageNet datasets.

AUTHOR:
    Mintae Kim
"""

import yaml

import torchvision
from torchvision.transforms import transforms as T

with open("config/loader.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def load():
    path = config['loader']['ImageNet']['path']

    # Dataset Preprocessing
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = torchvision.datasets.ImageNet(".", split="train", transform=transform)