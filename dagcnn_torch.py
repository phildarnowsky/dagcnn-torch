from random import choice, randint

import torch
from torch import nn
from torch.nn import functional
#   SEP3 = auto()
#   SEP5 = auto()
#   POOL = auto()
#   SUM = auto()
#   STACK = auto()

class AutoRepr():
    def __repr__(self):
        name = self.__class__.__name__
        attributes = str(self.__dict__)
        return f"<{name} {attributes}>"

class Node(AutoRepr):
    def to_block(self):
        raise NotImplementedError

    def output_feature_depth(self):
        raise NotImplementedError

    @classmethod
    def arity(cls):
        raise NotImplementedError

    @classmethod
    def make_random(cls):
        raise NotImplementedError

class Block(nn.Module):
    def __init__(self, predecessor_indices, output_feature_depth):
        super().__init__()
        self.predecessor_indices = predecessor_indices
        self.output_feature_depth = output_feature_depth
        self.net = None

    def forward(self, input):
        return self.net(input)

class ConvNode(Node):
    def __init__(self, output_feature_depth, kernel_size):
        super().__init__()
        self.__output_feature_depth = output_feature_depth
        self.kernel_size = kernel_size

    def to_block(self, predecessor_indices, input_feature_depth):
        return ConvBlock(predecessor_indices, input_feature_depth, self.output_feature_depth(), self.kernel_size)

    def output_feature_depth(self):
        return self.__output_feature_depth

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def make_random(cls):
        kernel_size = choice([1, 3])
        feature_depth = choice([32, 64, 128, 256])
        return cls(feature_depth, kernel_size)

class ConvBlock(Block):
    def __init__(self, predecessor_indices, input_feature_depth, output_feature_depth, kernel_size):
        super().__init__(predecessor_indices, output_feature_depth)
        padding = kernel_size // 2
        conv_layer = nn.Conv2d(input_feature_depth, output_feature_depth, kernel_size, padding=padding)
        relu_layer = nn.ReLU()
        batch_norm_layer = nn.BatchNorm2d(output_feature_depth)
        self.net = nn.Sequential(conv_layer, relu_layer, batch_norm_layer).cuda()

class AvgPoolNode(Node):
    def to_block(self, predecessor_indices, input_feature_depth):
        return AvgPoolBlock(predecessor_indices, input_feature_depth)

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def make_random(cls):
        return cls()

class AvgPoolBlock(Block):
    def __init__(self, predecessor_indices, output_feature_depth):
        super().__init__(predecessor_indices, output_feature_depth)
        self.net = nn.AvgPool2d(2, 2).cuda()

class Gene(AutoRepr):
    def __init__(self, node, predecessor_indices):
        assert(node.arity() == len(predecessor_indices))
        self.node = node
        self.predecessor_indices = predecessor_indices

    def to_block(self, model_input_feature_depth, blocks):
        input_feature_depths = []
        for predecessor_index in self.predecessor_indices:
            if predecessor_index == -1:
                input_feature_depths.append(model_input_feature_depth)
            else:
                input_feature_depths.append(blocks[predecessor_index].output_feature_depth)
        input_feature_depth = max(input_feature_depths)
        return self.node.to_block(self.predecessor_indices, input_feature_depth)


class Genome(AutoRepr):
    def __init__(self, input_feature_depth, output_feature_depth, genes):
        self.input_feature_depth = input_feature_depth
        self.output_feature_depth = output_feature_depth
        self.genes = genes

    def to_individual(self):
        blocks = []
        output_indices = set(range(len(self.genes)))

        for gene in self.genes:
#           input_feature_depth = self.__input_feature_depth(gene)
#           block = gene.node.to_block(gene.predecessor_indices, input_feature_depth)
            block = gene.to_block(self.input_feature_depth, blocks)
            blocks.append(block)
            output_indices = output_indices.difference(set(gene.predecessor_indices))
        return Individual(blocks, output_indices, self.output_feature_depth)

#   def __input_feature_depth(self, gene):
#       input_feature_depths = []
#       for predecessor_index in gene.predecessor_indices:
#           if predecessor_index == -1:
#               input_feature_depths.append(self.input_feature_depth)
#           else:
#               input_feature_depths.append(self.genes[predecessor_index].node.output_feature_depth())
#       return max(input_feature_depths)

    @classmethod
    def make_random(cls, input_feature_depth, output_feature_depth, min_length, max_length):
        length = randint(min_length, max_length)
        genes = []
        for index in range(length):
            node_class = choice(cls.__instantiable_classes())
            node = node_class.make_random()

            predecessor_indices = []
            for _ in range(node.arity()):
                predecessor_indices.append(randint(-1, index - 1))

            gene = Gene(node, predecessor_indices)
            genes.append(gene)

        return cls(input_feature_depth, output_feature_depth, genes)

    @classmethod
    def __instantiable_classes(cls):
        return [ConvNode, AvgPoolNode]

class Individual(nn.Module, AutoRepr):
    def __init__(self, blocks, output_indices, output_feature_depth, final_layer = nn.Identity()):
        super().__init__()
        self.blocks = blocks
        self.output_indices = list(output_indices)
        self.output_feature_depth = output_feature_depth
        self.final_layer = final_layer
        self.tail = None

    def forward(self, model_input):
        results = []
        for block in self.blocks:
            inputs = self.__get_block_inputs(model_input, results, block)
            result = block(*inputs)
            results.append(result)
        output_results = list(map(lambda output_index: results[output_index], self.output_indices))
        output_results = self.__match_shapes(output_results)
        output = torch.zeros_like(output_results[0])
        for output_result in output_results:
            output += output_result
        output = torch.flatten(output, 1)
        if self.tail == None:
            self.__make_tail(output.size(1))
        output = self.tail(output)
        return output

    def __get_block_inputs(self, model_input, results, block):
        inputs = []
        for predecessor_index in block.predecessor_indices:
            if predecessor_index == -1:
                inputs.append(model_input)
            else:
                inputs.append(results[predecessor_index])
        return inputs

    def __match_shapes(self, tensors):
        feature_depths = map(lambda tensor: tensor.size(1), tensors)
        heights = map(lambda tensor: tensor.size(2), tensors)
        max_feature_depth = max(feature_depths)
        max_height = max(heights)
        result = []
        for tensor in tensors:
            channel_difference = max_feature_depth - tensor.size(1)
            channel_padding_front = channel_difference // 2
            channel_padding_back = channel_difference - channel_padding_front
            height_and_width_padding = (max_height - tensor.size(2)) // 2
            padding = (height_and_width_padding, height_and_width_padding, height_and_width_padding, height_and_width_padding, channel_padding_front, channel_padding_back)
            padded_tensor = functional.pad(tensor, padding)
            result.append(padded_tensor)
        return result

    def __make_tail(self, input_feature_depth):
        fc_layer = nn.Linear(input_feature_depth, self.output_feature_depth).cuda()
        self.tail = nn.Sequential(fc_layer, self.final_layer).cuda()
