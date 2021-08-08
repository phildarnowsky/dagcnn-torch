from .max_pool_block import MaxPoolBlock
from .pool_gene import PoolGene

class MaxPoolGene(PoolGene):
    def __init__(self, index):
        super().__init__(index)

    def _block_class(self):
        return MaxPoolBlock

    def _cache_node_type(self):
        return "M"

    def _cache_parameters(self):
        return []
