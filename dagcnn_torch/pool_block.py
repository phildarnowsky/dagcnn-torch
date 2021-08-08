from torch import nn

from .block import Block

class PoolBlock(Block):
    def __init__(self, input_indices, input_shape):
        super().__init__(input_indices)
        self._input_shape = input_shape

        if self._is_too_small(self._input_shape):
            self._net = nn.Identity().cuda()
        else:
            self._net = self._layer_class()(2, 2).cuda()

    def output_shape(self):
        if self._is_too_small(self._input_shape):
            return self._input_shape
        else:
            return self._input_shape[0], self._input_shape[1] // 2, self._input_shape[2] // 2

    def _layer_class(self):
        raise NotImplementedError

    def _is_too_small(self, shape):
        return shape[1] < 2 or shape[2] < 2
