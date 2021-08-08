from torch import nn
from torch.nn.init import kaiming_normal_

from .block import Block

class ConvBlock(Block):
    def __init__(self, input_indices, input_shape, output_feature_depth, kernel_size):
        super().__init__(input_indices)
        self._input_shape = input_shape
        self._output_feature_depth = output_feature_depth

        input_feature_depth = input_shape[0]
        padding = kernel_size // 2
        conv_layer = nn.Conv2d(input_feature_depth, output_feature_depth, kernel_size, padding=padding)
        kaiming_normal_(conv_layer.weight)
        relu_layer = nn.ReLU()
        batch_norm_layer = nn.BatchNorm2d(output_feature_depth)
        self._net = nn.Sequential(conv_layer, relu_layer, batch_norm_layer).cuda()

    def output_shape(self):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 


