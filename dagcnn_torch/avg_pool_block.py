from torch import nn

from .pool_block import PoolBlock

class AvgPoolBlock(PoolBlock):
    def _layer_class(self):
        return nn.AvgPool2d
