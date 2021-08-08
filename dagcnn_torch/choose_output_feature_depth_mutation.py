from random import choice

from .mutation import Mutation

class ChooseOutputFeatureDepthMutation(Mutation):
    @classmethod
    def apply(cls, gene, _):
        new_output_feature_depth = choice(gene._valid_output_feature_depths())
        new_gene = gene.copy()
        new_gene._output_feature_depth = new_output_feature_depth
        return [new_gene]
