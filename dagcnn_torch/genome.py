from copy import deepcopy
from random import randint, random

from .auto_repr import AutoRepr
from .gene import Gene
from .gene_picker import GenePicker
from .individual import Individual

class Genome(AutoRepr):
    def __init__(self, input_shape, output_feature_depth, genes):
        self.input_shape = input_shape
        self.output_feature_depth = output_feature_depth
        self.genes = genes

    def crossover(self, other):
        n_genes = len(self.genes)
        if n_genes > len(other.genes):
            return(other.crossover(self))

        start_index = randint(0, n_genes - 1)
        end_index = randint(start_index + 1, n_genes)

        if random() < 0.5:
            parent1 = self
            parent2 = other
        else:
            parent1 = other
            parent2 = self

        head = parent1.genes[0:start_index]
        middle = parent2.genes[start_index:end_index]
        tail = parent1.genes[end_index:]
        child_genes = head + middle + tail
        assert(len(child_genes) == len(parent1.genes))

        child = Genome(self.input_shape, self.output_feature_depth, child_genes)
        return(child)

    def to_cache_key(self):
        return "".join(map(lambda gene: f"({gene.to_cache_key()})", self.genes))

    def to_individual(self):
        blocks = []
        output_indices = set(range(len(self.genes)))
        output_shapes = []

        for gene in self.genes:
            block = gene.to_block(self.input_shape, output_shapes)
            blocks.append(block)
            output_shapes.append(block.output_shape())
            output_indices = output_indices.difference(set(gene.input_indices))
        return Individual(blocks, self.input_shape, output_indices, self.output_feature_depth)

    def apply_mutations(self, mutation_probability):
        genome_length = len(self.genes)
        new_genes = []
        input_adjustments = [0] * genome_length

        for source_gene_index, source_gene in enumerate(self.genes):
            replacement_genes = self._possibly_apply_mutation_to_gene(source_gene, source_gene_index, mutation_probability)
            replacement_genes = self._apply_input_adjustments(replacement_genes, input_adjustments)

            adjustment_change = len(replacement_genes) - 1
            for i in range(source_gene_index, genome_length):
                input_adjustments[i] += adjustment_change

            new_genes += replacement_genes

        if(new_genes == []):
            new_genes = self.genes

        return Genome(self.input_shape, self.output_feature_depth, new_genes)

    def _possibly_apply_mutation_to_gene(self, source_gene, source_gene_index, mutation_probability):
        if random() > mutation_probability:
            return [source_gene]
        return source_gene.apply_random_mutation(source_gene_index)

    def _apply_input_adjustments(self, genes, input_adjustments):
        new_genes = []

        for gene in genes:
            new_input_indices = []
            for input_index in gene.input_indices:
                if input_index == -1:
                    new_input_indices.append(-1)
                else:
                    new_input_indices.append(input_index + input_adjustments[input_index])

            new_gene = deepcopy(gene)
            new_gene.input_indices = new_input_indices
            new_genes.append(new_gene)
        return new_genes

    @classmethod
    def make_random(cls, model_input_shape, model_output_feature_depth, min_length, max_length):
        length = randint(min_length, max_length)
        genes = []
        gene_picker = GenePicker()
        for index in range(length):
            gene = Gene.make_random(index, gene_picker)
            genes.append(gene)

        return cls(model_input_shape, model_output_feature_depth, genes)


