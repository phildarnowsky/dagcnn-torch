from random import choice, randint

from .auto_repr import AutoRepr
from .choose_input_mutation import ChooseInputMutation
from .deletion_mutation import DeletionMutation
from .insertion_mutation import InsertionMutation

class Gene(AutoRepr):
    def apply_random_mutation(self, source_gene_index):
        mutation = choice(self._valid_mutations())
        return mutation.apply(self, source_gene_index)

    def copy(self):
        raise NotImplementedError

    def to_cache_key(self):
        parameter_string = ",".join(map(str, self._cache_parameters()))
        input_indices_string = ",".join(map(str, self.input_indices))
        return "_".join([
            self._cache_node_type(),
            parameter_string,
            input_indices_string
        ])

    def output_shape(self):
        raise NotImplementedError

    def _cache_node_type(self):
        raise NotImplementedError

    def _cache_parameters(self):
        raise NotImplementedError

    def _get_input_shapes(self, model_input_shape, layer_output_shapes):
        return list(
            map(
                lambda input_index: self._get_input_shape(input_index, model_input_shape, layer_output_shapes),
                self.input_indices
            )
        )

    def _get_input_shape(self, input_index, model_input_shape, layer_output_shapes):
        if input_index == -1:
            return model_input_shape
        else:
            return layer_output_shapes[input_index]

    def _valid_mutations(self):
        basic_mutations = [DeletionMutation, InsertionMutation, ChooseInputMutation]
        return basic_mutations + self._class_specific_mutations()

    def _class_specific_mutations(self):
        return []

    @classmethod
    def arity(cls):
        raise NotImplementedError

    @classmethod
    def make_random(cls, index, gene_class_picker):
        gene_class = gene_class_picker.pick()
        return gene_class.make_random(index)

    @classmethod
    def _choose_input_indices(cls, index):
        input_indices = []
        for _ in range(cls.arity()):
            new_input_index = randint(-1, index - 1)
            input_indices.append(new_input_index)
        return input_indices
