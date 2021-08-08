from torch import nn

from .pool_block import PoolBlock

class MaxPoolBlock(PoolBlock):
    def _layer_class(self):
        return nn.MaxPool2d
