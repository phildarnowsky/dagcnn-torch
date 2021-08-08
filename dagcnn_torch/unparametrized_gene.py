from .gene import Gene

class UnparametrizedGene(Gene):
    def __init__(self, input_indices):
        self.input_indices = input_indices

    def copy(self):
        return self.__class__(self.input_indices)
