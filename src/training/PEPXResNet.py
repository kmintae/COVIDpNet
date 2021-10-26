"""PEPXResNet.py

AUTHOR:
    Mintae Kim

CITATIONS:
    Zhang, Y.; Zhang, Y. Deep Learning for COVID-19 Recognition. Preprints 2021, 2021050711 (doi: 10.20944/preprints202105.0711.v1).
    https://www.preprints.org/manuscript/202105.0711/v1
"""

import yaml

import torch
from torch import nn

import LearnableModels
import src.loader.ImageNet as ImageNet

with open("config/models.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['PEPX-ResNet']

class PEPX(nn.Module):
    def __init__(self):
        super(PEPX, self).__init__()

        divisor = config['parameters']['projection']
        multiplier = config['parameters']['expansion']

        # self.projection1 = nn.Conv2d()
        pass

    def forward(self, x):
        return x

class PEPXBlock(nn.Module):
    def __init__(self):
        super(PEPXBlock, self).__init__()

        # Extra Parameters: A, B, C
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(1))

        # Neural Network
        self.PEPX1 = PEPX()
        self.PEPX2 = PEPX()
        pass

    def forward(self, x):
        a = self.a.expand_as(x)
        b = self.b.expand_as(x)
        c = self.c.expand_as(x)

        # Neural Network
        x = self.PEPX(x)

        return x

class PEPXResNet(LearnableModels.LearnableModels, nn.Module):
    def __init__(self):
        super(PEPXResNet, self).__init__()

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
