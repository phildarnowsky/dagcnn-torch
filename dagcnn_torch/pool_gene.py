from .unparametrized_gene import UnparametrizedGene

class PoolGene(UnparametrizedGene):
    def to_block(self, model_input_shape, layer_output_shapes):
        input_shape = self._get_input_shape(self.input_indices[0], model_input_shape, layer_output_shapes)
        return self._block_class()(self.input_indices, input_shape)

    def _block_class(self):
        raise NotImplementedError

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def make_random(cls, index):
        input_indices = cls._choose_input_indices(index)
        return cls(input_indices)
