from copy import deepcopy
from datetime import datetime
from math import floor
from itertools import chain
from pickle import dump
from random import choice, randint, random

import torch
from torch import nn
from torch.nn import functional
from torch.nn.init import kaiming_normal_
from torch.optim import Adam

class AutoRepr():
    def __repr__(self):
        name = self.__class__.__name__
        attributes = str(self.__dict__)
        return f"<{name} {attributes}>"

class Gene(AutoRepr):
    def apply_random_mutation(self, source_gene_index):
        mutation = choice(self._valid_mutations())
        return mutation.apply(self, source_gene_index)

    def copy(self):
        raise NotImplementedError

    def to_cache_key(self):
        parameter_string = ",".join(map(str, self._cache_parameters()))
        input_indices_string = ",".join(map(str, self.input_indices))
        return "_".join([
            self._cache_node_type(),
            parameter_string,
            input_indices_string
        ])

    def output_shape(self):
        raise NotImplementedError

    def _cache_node_type(self):
        raise NotImplementedError

    def _cache_parameters(self):
        raise NotImplementedError

    def _get_input_shapes(self, model_input_shape, layer_output_shapes):
        return list(
            map(
                lambda input_index: self._get_input_shape(input_index, model_input_shape, layer_output_shapes),
                self.input_indices
            )
        )

    def _get_input_shape(self, input_index, model_input_shape, layer_output_shapes):
        if input_index == -1:
            return model_input_shape
        else:
            return layer_output_shapes[input_index]

    def _valid_mutations(self):
        basic_mutations = [DeletionMutation, InsertionMutation, ChooseInputMutation]
        return basic_mutations + self._class_specific_mutations()

    def _class_specific_mutations(self):
        return []

    @classmethod
    def arity(cls):
        raise NotImplementedError

    @classmethod
    def instantiable_classes(cls):
        return [ConvGene, DepSepConvGene, AvgPoolGene, MaxPoolGene, CatGene, SumGene]

    @classmethod
    def make_random(cls, index):
        gene_class = choice(Gene.instantiable_classes())
        return gene_class.make_random(index)

    @classmethod
    def _choose_input_indices(cls, index):
        input_indices = []
        for _ in range(cls.arity()):
            new_input_index = randint(-1, index - 1)
            input_indices.append(new_input_index)
        return input_indices

class Block(nn.Module):
    def __init__(self, input_indices):
        super().__init__()
        self.input_indices = input_indices
        self._net = lambda: None

    def forward(self, input):
        return self._net(input)

class AbstractConvGene(Gene):
    def __init__(self, input_indices, output_feature_depth, kernel_size):
        self.input_indices = input_indices
        self._output_feature_depth = output_feature_depth
        self._kernel_size = kernel_size

    def copy(self):
        return self.__class__(self.input_indices, self._output_feature_depth, self._kernel_size)

    def _class_specific_mutations(self):
        return [ChooseKernelSizeMutation, ChooseOutputFeatureDepthMutation]

    @classmethod
    def make_random(cls, index):
        input_indices = cls._choose_input_indices(index)
        kernel_size = choice(cls._valid_kernel_sizes())
        output_feature_depth = choice(cls._valid_output_feature_depths())
        return cls(input_indices, output_feature_depth, kernel_size)

    @classmethod
    def _valid_kernel_sizes(cls):
        raise NotImplementedError

    @classmethod
    def _valid_output_feature_depths(cls):
        return [32, 64, 128, 256]

class ConvGene(AbstractConvGene):
    def to_block(self, model_input_shape, layer_output_shapes):
        input_shape = self._get_input_shape(self.input_indices[0], model_input_shape, layer_output_shapes)
        return ConvBlock(self.input_indices, input_shape, self._output_feature_depth, self._kernel_size)

    def output_shape(self):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 

    def _cache_node_type(self):
        return "C"

    def _cache_parameters(self):
        return [self._kernel_size, self._output_feature_depth]

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def _valid_kernel_sizes(cls):
        return([1, 3])

class ConvBlock(Block):
    def __init__(self, input_indices, input_shape, output_feature_depth, kernel_size):
        super().__init__(input_indices)
        self._input_shape = input_shape
        self._output_feature_depth = output_feature_depth

        input_feature_depth = input_shape[0]
        padding = kernel_size // 2
        conv_layer = nn.Conv2d(input_feature_depth, output_feature_depth, kernel_size, padding=padding)
        kaiming_normal_(conv_layer.weight)
        relu_layer = nn.ReLU()
        batch_norm_layer = nn.BatchNorm2d(output_feature_depth)
        self._net = nn.Sequential(conv_layer, relu_layer, batch_norm_layer).cuda()

    def output_shape(self):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 

class DepSepConvGene(AbstractConvGene):
    def to_block(self, model_input_shape, layer_output_shapes):
        input_shape = self._get_input_shape(self.input_indices[0], model_input_shape, layer_output_shapes)
        return DepSepConvBlock(self.input_indices, input_shape, self._output_feature_depth, self._kernel_size)

    def output_shape(self):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 

    def _cache_node_type(self):
        return "D"

    def _cache_parameters(self):
        return [self._kernel_size, self._output_feature_depth]

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def _valid_kernel_sizes(cls):
        return([3, 5])

class DepSepConvBlock(Block):
    def __init__(self, input_indices, input_shape, output_feature_depth, kernel_size):
        super().__init__(input_indices)
        self._input_shape = input_shape
        self._output_feature_depth = output_feature_depth

        input_feature_depth = input_shape[0]
        padding = kernel_size // 2
        depthwise_layer = nn.Conv2d(input_feature_depth, input_feature_depth, kernel_size, groups=input_feature_depth, padding=padding)
        pointwise_layer = nn.Conv2d(input_feature_depth, output_feature_depth, 1)
        kaiming_normal_(depthwise_layer.weight)
        kaiming_normal_(pointwise_layer.weight)
        relu_layer = nn.ReLU()
        batch_norm_layer = nn.BatchNorm2d(output_feature_depth)
        self._net = nn.Sequential(depthwise_layer, pointwise_layer, relu_layer, batch_norm_layer).cuda()

    def output_shape(self):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 

class UnparametrizedGene(Gene):
    def __init__(self, input_indices):
        self.input_indices = input_indices

    def copy(self):
        return self.__class__(self.input_indices)

class PoolGene(UnparametrizedGene):
    def to_block(self, model_input_shape, layer_output_shapes):
        input_shape = self._get_input_shape(self.input_indices[0], model_input_shape, layer_output_shapes)
        return self._block_class()(self.input_indices, input_shape)

    def _block_class(self):
        raise NotImplementedError

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def make_random(cls, index):
        input_indices = cls._choose_input_indices(index)
        return cls(input_indices)

class PoolBlock(Block):
    def __init__(self, input_indices, input_shape):
        super().__init__(input_indices)
        self._input_shape = input_shape

        if self._is_too_small(self._input_shape):
            self._net = nn.Identity().cuda()
        else:
            self._net = self._layer_class()(2, 2).cuda()

    def output_shape(self):
        if self._is_too_small(self._input_shape):
            return self._input_shape
        else:
            return self._input_shape[0], self._input_shape[1] // 2, self._input_shape[2] // 2

    def _layer_class(self):
        raise NotImplementedError

    def _is_too_small(self, shape):
        return shape[1] < 2 or shape[2] < 2

class AvgPoolGene(PoolGene):
    def __init__(self, index):
        super().__init__(index)

    def _block_class(self):
        return AvgPoolBlock

    def _cache_node_type(self):
        return "A"

    def _cache_parameters(self):
        return []

class AvgPoolBlock(PoolBlock):
    def _layer_class(self):
        return nn.AvgPool2d

class MaxPoolGene(PoolGene):
    def __init__(self, index):
        super().__init__(index)

    def _block_class(self):
        return MaxPoolBlock

    def _cache_node_type(self):
        return "M"

    def _cache_parameters(self):
        return []

class MaxPoolBlock(PoolBlock):
    def _layer_class(self):
        return nn.MaxPool2d

class CatGene(UnparametrizedGene):
    def to_block(self, model_input_shape, layer_output_shapes):
        input_shapes = self._get_input_shapes(model_input_shape, layer_output_shapes)
        return CatBlock(self.input_indices, input_shapes)

    def _cache_node_type(self):
        return "K"

    def _cache_parameters(self):
        return []

    @classmethod
    def arity(cls):
        return 2

    @classmethod
    def make_random(cls, index):
        input_indices = cls._choose_input_indices(index)
        return cls(input_indices)

class CatBlock(Block):
    def __init__(self, input_indices, input_shapes):
        super().__init__(input_indices)
        self._input_shapes = input_shapes

    def output_shape(self):
        output_feature_depth = sum(map(lambda t: t[0], self._input_shapes))
        output_height = max(map(lambda t: t[1], self._input_shapes))
        output_width = max(map(lambda t: t[2], self._input_shapes))
        return (output_feature_depth, output_height, output_width)

    def forward(self, input1, input2):
        [padded_input1, padded_input2] = match_shapes([input1, input2], match_channels=False)
        return torch.cat((padded_input1, padded_input2), 1)

class SumGene(UnparametrizedGene):
    def to_block(self, model_input_shape, layer_output_shapes):
        input_shapes = self._get_input_shapes(model_input_shape, layer_output_shapes)
        return SumBlock(self.input_indices, input_shapes)

    def _cache_node_type(self):
        return "S"

    def _cache_parameters(self):
        return []

    @classmethod
    def arity(cls):
        return 2

    @classmethod
    def make_random(cls, index):
        input_indices = cls._choose_input_indices(index)
        return cls(input_indices)

class SumBlock(Block):
    def __init__(self, input_indices, input_shapes):
        super().__init__(input_indices)
        self._input_shapes = input_shapes

    def forward(self, input1, input2):
        [padded_input1, padded_input2] = match_shapes([input1, input2])
        return padded_input1 + padded_input2

    def output_shape(self):
        output_feature_depth = max(map(lambda t: t[0], self._input_shapes))
        output_height = max(map(lambda t: t[1], self._input_shapes))
        output_width = max(map(lambda t: t[2], self._input_shapes))
        return (output_feature_depth, output_height, output_width)

class Mutation():
    @classmethod
    def apply(cls, _gene, _index):
        raise NotImplementedError

class DeletionMutation(Mutation):
    @classmethod
    def apply(cls, _gene, _index):
        return []

class InsertionMutation(Mutation):
    @classmethod
    def apply(cls, gene, index):
        if random() < 0.5:
            return [gene, Gene.make_random(index + 1)]
        else:
            return [Gene.make_random(index), gene]

class ChooseInputMutation(Mutation):
    @classmethod
    def apply(cls, gene, index):
        new_input_index = randint(-1, index - 1)
        position_to_replace = randint(0, gene.arity() - 1)
        new_gene = gene.copy()
        new_gene.input_indices[position_to_replace] = new_input_index
        return [new_gene]

class ChooseKernelSizeMutation(Mutation):
    @classmethod
    def apply(cls, gene, _):
        new_kernel_size = choice(gene._valid_kernel_sizes())
        new_gene = gene.copy()
        new_gene._kernel_size = new_kernel_size
        return [new_gene]

class ChooseOutputFeatureDepthMutation(Mutation):
    @classmethod
    def apply(cls, gene, _):
        new_output_feature_depth = choice(gene._valid_output_feature_depths())
        new_gene = gene.copy()
        new_gene._output_feature_depth = new_output_feature_depth
        return [new_gene]

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
        for index in range(length):
            gene = Gene.make_random(index)
            genes.append(gene)

        return cls(model_input_shape, model_output_feature_depth, genes)

class Individual(nn.Module, AutoRepr):
    def __init__(self, blocks, input_shape, output_indices, output_feature_depth, final_layer = nn.Identity()):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.output_indices = list(output_indices)
        self.output_feature_depth = output_feature_depth
        self.tail = self._make_tail(input_shape, final_layer)

    def forward(self, model_input):
        output_results = self._calculate_output_results(model_input)
        output_results = match_shapes(output_results)
        output = torch.zeros_like(output_results[0])
        for output_result in output_results:
            output += output_result
        output = self.tail(output)
        return output

    def parameters(self):
        return chain(self.blocks.parameters(), self.tail.parameters())

    def _get_block_inputs(self, model_input, results, block):
        inputs = []
        for input_index in block.input_indices:
            if input_index == -1:
                inputs.append(model_input.cuda())
            else:
                inputs.append(results[input_index].cuda())
        return inputs

    def _make_tail(self, input_shape, final_layer):
        with torch.no_grad():
            self.eval()
            dummy_input = torch.zeros(1, *input_shape).cuda()
            dummy_output_results = self._calculate_output_results(dummy_input)
            output_shape = largest_dimensions(dummy_output_results)
        (input_feature_depth, height, width) = output_shape
        gap_layer = nn.AvgPool2d(kernel_size=(height, width)).cuda()
        flatten_layer = nn.Flatten()
        fc_layer = nn.Linear(input_feature_depth, self.output_feature_depth).cuda()
        kaiming_normal_(fc_layer.weight)
        return nn.Sequential(gap_layer, flatten_layer, fc_layer, final_layer).cuda()

    def _calculate_output_results(self, model_input):
        results = []
        for _, block in enumerate(self.blocks):
            inputs = self._get_block_inputs(model_input, results, block)
            result = block(*inputs).cpu()
            results.append(result)
        return list(map(lambda output_index: results[output_index].cuda(), self.output_indices))

def largest_dimensions(tensors):
    feature_depths = map(lambda tensor: tensor.size(1), tensors)
    heights = map(lambda tensor: tensor.size(2), tensors)
    widths = map(lambda tensor: tensor.size(3), tensors)
    max_feature_depth = max(feature_depths)
    max_height = max(heights)
    max_width = max(widths)
    return(max_feature_depth, max_height, max_width)

def match_shapes(tensors, match_channels=True):
    (max_feature_depth, max_height, max_width) = largest_dimensions(tensors)
    result = []
    for tensor in tensors:
        if match_channels:
            channel_difference = max_feature_depth - tensor.size(1)
        else:
            channel_difference = 0
        channel_padding_front = channel_difference // 2
        channel_padding_back = channel_difference - channel_padding_front
        
        height_difference = max_height - tensor.size(2)
        height_padding_top = height_difference // 2
        height_padding_bottom = height_difference - height_padding_top
        
        width_difference = max_width - tensor.size(3)
        width_padding_left = width_difference // 2
        width_padding_right = width_difference - width_padding_left

        padding = (width_padding_left, width_padding_right, height_padding_top, height_padding_bottom, channel_padding_front, channel_padding_back)
        padded_tensor = functional.pad(tensor, padding)
        result.append(padded_tensor)
    return result

class Population():
    def __init__(
        self,
        genomes,
        training_loader,
        validation_loader,
        make_optimizer=lambda individual: Adam(individual.parameters()),
        criterion_class=nn.CrossEntropyLoss,
        elitism_fraction=0.2,
        mean_threshold=0.2,
        std_threshold=0.02,
        mutation_probability=0.003
    ):
        self._genomes = genomes
        n_genomes = len(genomes)
        self._n_by_elitism = floor(elitism_fraction * n_genomes)
        self._n_by_breeding = n_genomes - self._n_by_elitism

        self._fitness_cache = {}
        self._training_loader = training_loader
        self._validation_loader = validation_loader
        self._make_optimizer = make_optimizer
        self._criterion_class = criterion_class
        self._mean_threshold = mean_threshold
        self._std_threshold = std_threshold
        self._mutation_probability = mutation_probability

    def breed_next_generation(self):
        new_genomes = []
        for _ in range(0, self._n_by_elitism):
            elite_genome = self._select_by_slack_binary_tournament()
            new_genomes.append(elite_genome)

        for _ in range(0, self._n_by_breeding):
            parent1 = self._select_by_slack_binary_tournament()
            parent2 = self._select_by_slack_binary_tournament()
            child = parent1.crossover(parent2)
            new_genomes.append(child)

        new_genomes = list(
            map(
                lambda genome: genome.apply_mutations(self._mutation_probability),
                new_genomes
            )
        )
        self._genomes = new_genomes

    def all_fitnesses(self):
        return {genome : self._fitness(genome) for genome in self._genomes}

    def _fitness(self, genome):
        cache_key = genome.to_cache_key()
        if cache_key not in self._fitness_cache:
            self._fitness_cache[cache_key] = self._evaluate_fitness(genome)
        return self._fitness_cache[cache_key]

    def _evaluate_fitness(self, genome):
        individual = genome.to_individual()
        validation_losses = []
        criterion = self._criterion_class()
        optimizer = self._make_optimizer(individual)

        individual.train()
        for _, (training_examples, training_labels) in enumerate(self._training_loader):
            training_examples = training_examples.cuda()
            training_labels = training_labels.cuda()
            training_predictions = individual(training_examples)

            training_loss = criterion(training_predictions, training_labels.flatten())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        individual.eval()
        with torch.no_grad():
            for _, (validation_examples, validation_labels) in enumerate(self._validation_loader):
                validation_examples = validation_examples.cuda()
                validation_labels = validation_labels.cuda()
                validation_predictions = individual(validation_examples)

                validation_loss = criterion(validation_predictions, validation_labels.flatten())
                validation_losses.append(validation_loss.item())

        validation_losses = torch.tensor(validation_losses)
        n_parameters = sum(list(map(lambda parameter: parameter.size().numel(), individual.parameters())))

        return {'mean': validation_losses.mean().item(), 'std': validation_losses.std().item(), 'n_parameters': n_parameters}

    def _select_by_slack_binary_tournament(self):
        genome1 = choice(self._genomes)
        genome2 = choice(self._genomes)

        fitness1 = self._fitness(genome1)
        fitness2 = self._fitness(genome2)

        if fitness1['mean'] - fitness2['mean'] > self._mean_threshold:
            return genome2
        elif fitness2['mean'] - fitness1['mean'] > self._mean_threshold:
            return genome1
        elif fitness1['std'] - fitness2['std'] > self._std_threshold:
            return genome2
        elif fitness2['std'] - fitness1['std'] > self._std_threshold:
            return genome1
        elif fitness1['n_parameters'] < fitness2['n_parameters']:
            return genome1

        return genome2

    @classmethod
    def make_random(
        cls,
        n_genomes,
        input_shape,
        n_outputs,
        minimum_length,
        maximum_length,
        training_loader,
        validation_loader,
        make_optimizer=lambda individual: Adam(individual.parameters()),
        criterion_class=nn.CrossEntropyLoss,
        elitism_fraction=0.2,
        mutation_probability=0.003
    ):
        genomes = []
        for _ in range(n_genomes):
            genomes.append(Genome.make_random(input_shape, n_outputs, minimum_length, maximum_length))
        return cls(genomes, training_loader, validation_loader, make_optimizer, criterion_class, elitism_fraction=elitism_fraction, mutation_probability=mutation_probability)

def saynow(text):
    print(f"[{datetime.now()}] {text}") 

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = 10
    n_genomes = 100
    min_n_genes = 3
    max_n_genes = 5
    n_generations = 100
    elitism_fraction = 0.2
    #elitism_fraction = 0
    mutation_probability = 0.003
    #mutation_probability = 0.5
    #mutation_probability = 100

    full_training_data = torch.load("./datasets/cifar-10/raw/all_training_data.pt").to(dtype=torch.float32)
    full_training_data_mean = full_training_data.mean()
    full_training_data_std = full_training_data.std()

    training_data = torch.load("./datasets/cifar-10/processed/training_data.pt").to(dtype=torch.float32)
    training_data = (training_data - full_training_data_mean) / full_training_data_std
    training_labels = torch.load("./datasets/cifar-10/processed/training_labels.pt").to(dtype=torch.long)
    training_dataset = TensorDataset(training_data, training_labels)
    training_loader = DataLoader(training_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

    validation_data = torch.load("./datasets/cifar-10/processed/validation_data.pt").to(dtype=torch.float32)
    validation_data = (validation_data - full_training_data_mean) / full_training_data_std
    validation_labels = torch.load("./datasets/cifar-10/processed/validation_labels.pt").to(dtype=torch.long)
    validation_dataset = TensorDataset(validation_data, validation_labels)
    validation_loader = DataLoader(validation_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

    population = Population.make_random(n_genomes, (3, 32, 32), 10, min_n_genes, max_n_genes, training_loader, validation_loader, elitism_fraction=elitism_fraction, mutation_probability=mutation_probability)

    saynow(list(map(lambda genome: genome.to_cache_key(), population._genomes)))
    for i in range(n_generations):
        saynow(f"GENERATION {i}")
        population.breed_next_generation()
    saynow(list(map(lambda genome: genome.to_cache_key(), population._genomes)))

    saynow("COMPUTING ALL FITNESSES FOR FINAL GENERATION")
    final_fitnesses = population.all_fitnesses()
    saynow("AND DONE!")

    dump_filename = f"./experiment_results/cifar_10_classifier_{datetime.now().isoformat()}.pickle"
    with open(dump_filename, "wb") as f:
        dump(final_fitnesses, f)
