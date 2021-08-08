from .avg_pool_block import AvgPoolBlock
from .pool_gene import PoolGene

class AvgPoolGene(PoolGene):
    def __init__(self, index):
        super().__init__(index)

    def _block_class(self):
        return AvgPoolBlock

    def _cache_node_type(self):
        return "A"

    def _cache_parameters(self):
        return []
