"""LearnableModels.py

Interfaces for 'Strategy' design pattern.

AUTHOR:
    Mintae Kim
"""

import abc

class LearnableModels:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def pretrain(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def test(self):
        pass

