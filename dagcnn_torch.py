from datetime import datetime
from itertools import chain
from random import choice, randint

import torch
from torch import nn
from torch.nn import functional
from torch.nn.init import kaiming_normal_

class AutoRepr():
    def __repr__(self):
        name = self.__class__.__name__
        attributes = str(self.__dict__)
        return f"<{name} {attributes}>"

class Node(AutoRepr):
    def __init__(self, input_shapes):
        self.input_shapes = input_shapes

    def to_block(self, _):
        raise NotImplementedError

    def to_cache_key(self):
        return "_".join([
            self.cache_node_type(),
            ",".join(map(str, self.cache_parameters()))
        ])

    def output_shape(self, _):
        raise NotImplementedError

    def cache_node_type(self):
        raise NotImplementedError

    def cache_parameters(self):
        raise NotImplementedError

    @classmethod
    def arity(cls):
        raise NotImplementedError

    @classmethod
    def make_random(cls, _):
        raise NotImplementedError

class Block(nn.Module):
    def __init__(self, input_indices):
        super().__init__()
        self.input_indices = input_indices
        self.net = None

    def forward(self, input):
        return self.net(input)

class ConvNode(Node):
    def __init__(self, input_shapes, output_feature_depth, kernel_size):
        super().__init__(input_shapes)
        self._input_shape = self.input_shapes[0]
        self._output_feature_depth = output_feature_depth
        self._kernel_size = kernel_size

    def to_block(self, input_indices):
        return ConvBlock(input_indices, self._input_shape[0], self._output_feature_depth, self._kernel_size)

    def output_shape(self, _):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 

    def cache_node_type(self):
        return "C"

    def cache_parameters(self):
        return [self._kernel_size, self._output_feature_depth]

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def make_random(cls, input_shapes):
        kernel_size = choice([1, 3])
        output_feature_depth = choice([32, 64, 128, 256])
        return cls(input_shapes, output_feature_depth, kernel_size)

class ConvBlock(Block):
    def __init__(self, input_indices, input_feature_depth, output_feature_depth, kernel_size):
        super().__init__(input_indices)
        padding = kernel_size // 2
        conv_layer = nn.Conv2d(input_feature_depth, output_feature_depth, kernel_size, padding=padding)
        kaiming_normal_(conv_layer.weight)
        relu_layer = nn.ReLU()
        batch_norm_layer = nn.BatchNorm2d(output_feature_depth)
        self.net = nn.Sequential(conv_layer, relu_layer, batch_norm_layer).cuda()

class DepSepConvNode(Node):
    def __init__(self, input_feature_depths, output_feature_depth, kernel_size):
        super().__init__(input_feature_depths)
        self._input_shape = self.input_shapes[0]
        self._output_feature_depth = output_feature_depth
        self._kernel_size = kernel_size

    def to_block(self, input_indices):
        return DepSepConvBlock(input_indices, self._input_shape[0], self._output_feature_depth, self._kernel_size)

    def output_shape(self, _):
        return (self._output_feature_depth, self._input_shape[1], self._input_shape[2]) 

    def cache_node_type(self):
        return "D"

    def cache_parameters(self):
        return [self._kernel_size, self._output_feature_depth]

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def make_random(cls, input_shapes):
        kernel_size = choice([3, 5])
        output_feature_depth = choice([32, 64, 128, 256])
        return cls(input_shapes, output_feature_depth, kernel_size)

class DepSepConvBlock(Block):
    def __init__(self, input_indices, input_feature_depth, output_feature_depth, kernel_size):
        super().__init__(input_indices)
        padding = kernel_size // 2
        depthwise_layer = nn.Conv2d(input_feature_depth, input_feature_depth, kernel_size, groups=input_feature_depth, padding=padding)
        pointwise_layer = nn.Conv2d(input_feature_depth, output_feature_depth, 1)
        kaiming_normal_(depthwise_layer.weight)
        kaiming_normal_(pointwise_layer.weight)
        relu_layer = nn.ReLU()
        batch_norm_layer = nn.BatchNorm2d(output_feature_depth)
        self.net = nn.Sequential(depthwise_layer, pointwise_layer, relu_layer, batch_norm_layer).cuda()

class PoolNode(Node):
    def is_too_small(self, shape):
        return shape[1] < 2 or shape[2] < 2

    def output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        if self.is_too_small(input_shape):
            return input_shape
        else:
            return input_shape[0], input_shape[1] // 2, input_shape[2] // 2

    def to_block(self, input_indices):
        return self.block_class()(input_indices, self.is_too_small(self.input_shapes[0]))

    def block_class(self):
        raise NotImplementedError

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def make_random(cls, input_shapes):
        return cls(input_shapes)

class PoolBlock(Block):
    def __init__(self, input_indices, is_too_small):
        super().__init__(input_indices)
        if is_too_small:
            self.net = nn.Identity().cuda()
        else:
            self.net = self.layer_class()(2, 2).cuda()

    def layer_class(self):
        raise NotImplementedError

class AvgPoolNode(PoolNode):
    def block_class(self):
        return AvgPoolBlock

    def cache_node_type(self):
        return "A"

    def cache_parameters(self):
        return []

class AvgPoolBlock(PoolBlock):
    def layer_class(self):
        return nn.AvgPool2d

class MaxPoolNode(PoolNode):
    def block_class(self):
        return MaxPoolBlock

    def cache_node_type(self):
        return "M"

    def cache_parameters(self):
        return []

class MaxPoolBlock(PoolBlock):
    def layer_class(self):
        return nn.MaxPool2d

class CatNode(Node):
    def to_block(self, input_indices):
        return CatBlock(input_indices)

    def output_shape(self, input_shapes):
        output_feature_depth = sum(map(lambda t: t[0], input_shapes))
        output_height = max(map(lambda t: t[0], input_shapes))
        output_width = max(map(lambda t: t[0], input_shapes))
        return (output_feature_depth, output_height, output_width)

    def cache_node_type(self):
        return "K"

    def cache_parameters(self):
        return []

    @classmethod
    def arity(cls):
        return 2

    @classmethod
    def make_random(cls, input_feature_depths):
        return cls(input_feature_depths)

class CatBlock(Block):
    def forward(self, input1, input2):
        [padded_input1, padded_input2] = match_shapes([input1, input2], match_channels=False)
        return torch.cat((padded_input1, padded_input2), 1)

class SumNode(Node):
    def to_block(self, input_indices):
        return SumBlock(input_indices)

    def output_shape(self, input_shapes):
        output_feature_depth = max(map(lambda t: t[0], input_shapes))
        output_height = max(map(lambda t: t[0], input_shapes))
        output_width = max(map(lambda t: t[0], input_shapes))
        return (output_feature_depth, output_height, output_width)

    def cache_node_type(self):
        return "S"

    def cache_parameters(self):
        return []

    @classmethod
    def arity(cls):
        return 2

    @classmethod
    def make_random(cls, input_feature_depths):
        return cls(input_feature_depths)

class SumBlock(Block):
    def forward(self, input1, input2):
        [padded_input1, padded_input2] = match_shapes([input1, input2])
        return padded_input1 + padded_input2
        
class Gene(AutoRepr):
    def __init__(self, node, input_indices, input_shapes):
        assert(node.arity() == len(input_indices))
        self.node = node
        self.input_indices = input_indices
        self.input_shapes = input_shapes

    def output_shape(self):
        return self.node.output_shape(self.input_shapes)

    def to_block(self):
        return self.node.to_block(self.input_indices)

    def to_cache_key(self):
        return "({node_cache_key}_{input_indices})".format(
            node_cache_key = self.node.to_cache_key(),
            input_indices = ",".join(map(str, self.input_indices))
        )

class Genome(AutoRepr):
    def __init__(self, input_shape, output_feature_depth, genes):
        self.input_shape = input_shape
        self.output_feature_depth = output_feature_depth
        self.genes = genes

    def to_cache_key(self):
        return "".join(map(lambda gene: gene.to_cache_key(), self.genes))

    def to_individual(self):
        blocks = []
        output_indices = set(range(len(self.genes)))

        for gene in self.genes:
            block = gene.to_block()
            blocks.append(block)
            output_indices = output_indices.difference(set(gene.input_indices))
        return Individual(blocks, self.input_shape, output_indices, self.output_feature_depth)

    @classmethod
    def make_random(cls, model_input_shape, model_output_feature_depth, min_length, max_length):
        length = randint(min_length, max_length)
        genes = []
        for index in range(length):
            node_class = choice(cls._instantiable_classes())
            input_indices = []
            input_feature_shapes = []
            for _ in range(node_class.arity()):
                new_input_index = randint(-1, index - 1)
                input_indices.append(new_input_index)
                if new_input_index == -1:
                    input_feature_shapes.append(model_input_shape)
                else:
                    input_feature_shapes.append(genes[new_input_index].output_shape())

            node = node_class.make_random(input_feature_shapes)

            gene = Gene(node, input_indices, input_feature_shapes)
            genes.append(gene)

        return cls(model_input_shape, model_output_feature_depth, genes)

    @classmethod
    def _instantiable_classes(cls):
        return [ConvNode, DepSepConvNode, AvgPoolNode, MaxPoolNode, CatNode, SumNode]

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
                inputs.append(model_input)
            else:
                inputs.append(results[input_index])
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
        for block in self.blocks:
            inputs = self._get_block_inputs(model_input, results, block)
            result = block(*inputs)
            results.append(result)
        return list(map(lambda output_index: results[output_index], self.output_indices))

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
    def __init__(self, genomes):
        self.genomes = genomes
        self.fitness_cache = {}

    @classmethod
    def make_random(cls, n_genomes, input_shape, n_outputs, minimum_length, maximum_length):
        genomes = []
        for _ in range(n_genomes):
            genomes.append(Genome.make_random(input_shape, n_outputs, minimum_length, maximum_length))
        return cls(genomes)

if __name__ == "__main__":
    from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
    from torch.optim import Adam

    #evolution_set_size = 4500
    evolution_set_size = 4
    n_epochs = 2
    n_genes = 25

    data = torch.load("./datasets/cifar-10/raw/all_training_data.pt").to(dtype=torch.float32)
    labels = torch.load("./datasets/cifar-10/raw/all_training_labels.pt")
    data = data[0:evolution_set_size]
    labels = labels[0:evolution_set_size]
    dataset = TensorDataset(data, labels)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, pin_memory=True)
    population = Population.make_random(1, (3, 32, 32), 10, n_genes, n_genes)
    genome_index = 0
    for genome in population.genomes:
        criterion = nn.CrossEntropyLoss()
        print(f"GENOME {genome_index}")
        print(genome.to_cache_key())
        individual = genome.to_individual()
        optimizer = Adam(individual.parameters())
        for epoch_index in range(n_epochs):
            print(f"[{datetime.now()}] {genome_index}/{epoch_index}")
            losses = []
            for element_index, [image, label] in enumerate(loader):
                prediction = individual(image.cuda())
                loss = criterion(prediction, label.cuda().flatten())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            average_loss = sum(losses) / len(losses)
            print(average_loss)
        genome_index += 1
