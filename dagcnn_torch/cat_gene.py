from .cat_block import CatBlock
from .unparametrized_gene import UnparametrizedGene

class CatGene(UnparametrizedGene):
    def to_block(self, model_input_shape, layer_output_shapes):
        input_shapes = self._get_input_shapes(model_input_shape, layer_output_shapes)
        return CatBlock(self.input_indices, input_shapes)

    def _cache_node_type(self):
        return "K"

    def _cache_parameters(self):
        return []

    @classmethod
    def arity(cls):
        return 2

    @classmethod
    def make_random(cls, index):
        input_indices = cls._choose_input_indices(index)
        return cls(input_indices)
