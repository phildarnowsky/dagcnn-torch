from random import choice

from .conv_gene import ConvGene
from .dep_sep_conv_gene import DepSepConvGene
from .cat_gene import CatGene
from .sum_gene import SumGene
from .avg_pool_gene import AvgPoolGene
from .max_pool_gene import MaxPoolGene

class GenePicker:
    def pick(self):
        return choice(self._instantiable_classes())
    
    def _instantiable_classes(self):
        return [ConvGene, DepSepConvGene, CatGene, SumGene, AvgPoolGene, MaxPoolGene]
