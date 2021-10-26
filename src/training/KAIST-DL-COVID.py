import yaml

import torch
from torch import nn

import LearnableModels
import src.loader.ImageNet as ImageNet

with open("config/models.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['KAIST-DLNet']

class KAISTDLNet(LearnableModels.LearnableModels, nn.Module):
    def __init__(self):
        super(KAISTDLNet, self).__init__()

        pass

    def forward(self, x):
        # Y

        pass

    def pretrain(self):
        dataset_imagenet = ImageNet.load()
        pass

    def train(self):
        pass

    def test(self):
        pass