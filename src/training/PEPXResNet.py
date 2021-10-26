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
    def __init__(self, in_features: int, out_features: int):
        super(PEPX, self).__init__()

        divisor = config['parameters']['projection']
        multiplier = config['parameters']['expansion']

        self.projection1 = nn.Conv2d() # 1x1 Conv
        self.expansion = nn.Conv2d() # 1x1 Conv
        self.representation = nn.Conv2d() # Depth-wise 3x3 Conv
        self.projection2 = nn.Conv2d() # 1x1 Conv
        self.extension = nn.Conv2d() # 1x1 Conv
        pass

    def forward(self, x):
        x = self.projection1(x)
        x = self.expansion(x)
        x = self.representation(x)
        x = self.projection2(x)
        x = self.extension(x)
        return x

class PEPXBlock(nn.Module):
    def __init__(self):
        super(PEPXBlock, self).__init__()

        # Extra Parameters: A, B, C
        self.a1 = nn.Parameter(torch.zeros(1))
        self.b1 = nn.Parameter(torch.zeros(1))
        self.c1 = nn.Parameter(torch.zeros(1))
        self.a2 = nn.Parameter(torch.zeros(1))
        self.b2 = nn.Parameter(torch.zeros(1))
        self.c2 = nn.Parameter(torch.zeros(1))
        self.d  = nn.Parameter(torch.zeros(1))

        # Neural Network
        self.PEPX1 = PEPX()
        self.PEPX2 = PEPX()

        # Same Dimension & Channel #
        self.f1_1 = nn.Conv2d()
        self.f1_2 = nn.Conv2d()
        self.f2_1 = nn.Conv2d()
        self.f2_2 = nn.Conv2d()
        pass

    def forward(self, x):
        # Neural Network
        x = self.PEPX(x)

        # Option 1: Y = F [F1(x) + x] + A * F1(X) + B * x + C * PEPX(X)
        # Option 2: Graphs (Selected)
        pepx_c = self.PEPX1(x)
        x1 = self.a1 * pepx_c + \
             self.b1 * x + \
             self.c1 * self.f1_2(self.f1_1(x))

        x = self.a2 * self.PEPX2(pepx_c) + \
            self.b2 * x1 + \
            self.c2 * self.f2_2(self.f2_1(x1)) + \
            self.d * x

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
