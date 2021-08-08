from torch import nn
from torch.nn.init import kaiming_normal_

from .block import Block

class DepSepConvBlock(Block):
    def __init__(self, input_indices, input_shape, output_feature_depth, kernel_size):
        super().__init__(input_indices)
        self._input_shape = input_shape
        self._output_feature_depth = output_feature_depth

        input_feature_depth = input_shape[0]
        padding = kernel_size // 2
        depthwise_layer = nn.Conv2d(input_feature_depth, input_feature_depth, kernel_size, groups=input_feature_depth, padding=padding)
        pointwise_layer = nn.Conv2d(input_feature_depth, output_feature_depth, 1)
        kaiming_normal_(depthwise_layer.weight)
        kaiming_normal_(pointwise_layer.weight)
        relu_layer = nn.ReLU()
        batch_norm_layer = nn.BatchNorm2d(output_feature_depth)
        self._net = nn.Sequential(depthwise_layer, pointwise_layer, relu_layer, batch_norm_layer).cuda()

    def output_shape(self):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 
