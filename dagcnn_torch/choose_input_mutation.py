from random import randint
from .mutation import Mutation

class ChooseInputMutation(Mutation):
    @classmethod
    def apply(cls, gene, index):
        new_input_index = randint(-1, index - 1)
        position_to_replace = randint(0, gene.arity() - 1)
        new_gene = gene.copy()
        new_gene.input_indices[position_to_replace] = new_input_index
        return [new_gene]
