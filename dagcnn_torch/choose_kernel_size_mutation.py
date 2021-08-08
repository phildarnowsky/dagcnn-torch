from random import choice

from .mutation import Mutation

class ChooseKernelSizeMutation(Mutation):
    @classmethod
    def apply(cls, gene, _):
        new_kernel_size = choice(gene._valid_kernel_sizes())
        new_gene = gene.copy()
        new_gene._kernel_size = new_kernel_size
        return [new_gene]
