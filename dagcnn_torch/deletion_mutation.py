from .mutation import Mutation

class DeletionMutation(Mutation):
    @classmethod
    def apply(cls, _gene, _index):
        return []
