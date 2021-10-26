import yaml

import torch
import torchvision

import LearnableModels
import src.loader.ImageNet as ImageNet

with open("config/models.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['CpNet']

class CpNet(LearnableModels.LearnableModels):
    def pretrain(self):
        dataset_imagenet = ImageNet.load()
        pass

    def train(self):
        pass

    def test(self):
        pass