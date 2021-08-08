from .block import Block
from .match_shapes import match_shapes

class SumBlock(Block):
    def __init__(self, input_indices, input_shapes):
        super().__init__(input_indices)
        self._input_shapes = input_shapes

    def forward(self, input1, input2):
        [padded_input1, padded_input2] = match_shapes([input1, input2])
        return padded_input1 + padded_input2

    def output_shape(self):
        output_feature_depth = max(map(lambda t: t[0], self._input_shapes))
        output_height = max(map(lambda t: t[1], self._input_shapes))
        output_width = max(map(lambda t: t[2], self._input_shapes))
        return (output_feature_depth, output_height, output_width)
