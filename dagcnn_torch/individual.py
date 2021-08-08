from itertools import chain

import torch
from torch import nn
from torch.nn.init import kaiming_normal_

from .auto_repr import AutoRepr
from .largest_dimensions import largest_dimensions
from .match_shapes import match_shapes

class Individual(nn.Module, AutoRepr):
    def __init__(self, blocks, input_shape, output_indices, output_feature_depth, final_layer = nn.Identity()):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.output_indices = list(output_indices)
        self.output_feature_depth = output_feature_depth
        self.tail = self._make_tail(input_shape, final_layer)

    def forward(self, model_input):
        output_results = self._calculate_output_results(model_input)
        output_results = match_shapes(output_results)
        output = torch.zeros_like(output_results[0])
        for output_result in output_results:
            output += output_result
        output = self.tail(output)
        return output

    def parameters(self):
        return chain(self.blocks.parameters(), self.tail.parameters())

    def _get_block_inputs(self, model_input, results, block):
        inputs = []
        for input_index in block.input_indices:
            if input_index == -1:
                inputs.append(model_input.cuda())
            else:
                inputs.append(results[input_index].cuda())
        return inputs

    def _make_tail(self, input_shape, final_layer):
        with torch.no_grad():
            self.eval()
            dummy_input = torch.zeros(1, *input_shape).cuda()
            dummy_output_results = self._calculate_output_results(dummy_input)
            output_shape = largest_dimensions(dummy_output_results)
        (input_feature_depth, height, width) = output_shape
        gap_layer = nn.AvgPool2d(kernel_size=(height, width)).cuda()
        flatten_layer = nn.Flatten()
        fc_layer = nn.Linear(input_feature_depth, self.output_feature_depth).cuda()
        kaiming_normal_(fc_layer.weight)
        return nn.Sequential(gap_layer, flatten_layer, fc_layer, final_layer).cuda()

    def _calculate_output_results(self, model_input):
        results = []
        for _, block in enumerate(self.blocks):
            inputs = self._get_block_inputs(model_input, results, block)
            result = block(*inputs).cpu()
            results.append(result)
        return list(map(lambda output_index: results[output_index].cuda(), self.output_indices))
