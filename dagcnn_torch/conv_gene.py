from .abstract_conv_gene import AbstractConvGene
from .conv_block import ConvBlock

class ConvGene(AbstractConvGene):
    def to_block(self, model_input_shape, layer_output_shapes):
        input_shape = self._get_input_shape(self.input_indices[0], model_input_shape, layer_output_shapes)
        return ConvBlock(self.input_indices, input_shape, self._output_feature_depth, self._kernel_size)

    def output_shape(self):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 

    def _cache_node_type(self):
        return "C"

    def _cache_parameters(self):
        return [self._kernel_size, self._output_feature_depth]

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def _valid_kernel_sizes(cls):
        return([1, 3])
