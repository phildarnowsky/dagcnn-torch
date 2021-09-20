from random import choice

from .gene import Gene
from .choose_kernel_size_mutation import ChooseKernelSizeMutation
from .choose_output_feature_depth_mutation import ChooseOutputFeatureDepthMutation

class AbstractConvGene(Gene):
    def __init__(self, input_indices, output_feature_depth, kernel_size):
        self.input_indices = input_indices
        self._output_feature_depth = output_feature_depth
        self._kernel_size = kernel_size

    def copy(self):
        return self.__class__(self.input_indices, self._output_feature_depth, self._kernel_size)

    def _class_specific_mutations(self):
        return [ChooseOutputFeatureDepthMutation]

    @classmethod
    def make_random(cls, index):
        input_indices = cls._choose_input_indices(index)
        kernel_size = choice(cls._valid_kernel_sizes())
        output_feature_depth = choice(cls._valid_output_feature_depths())
        return cls(input_indices, output_feature_depth, kernel_size)

    @classmethod
    def _valid_kernel_sizes(cls):
        raise NotImplementedError

    @classmethod
    def _valid_output_feature_depths(cls):
        return [32, 64, 128, 256]


