from random import random

from .mutation import Mutation

class InsertionMutation(Mutation):
    @classmethod
    def apply(cls, gene, index):
        if random() < 0.5:
            return [gene, gene.make_random(index + 1)]
        else:
            return [gene.make_random(index), gene]
