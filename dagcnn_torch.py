from random import choice, randint

from torch import nn
#   CONV1 = auto()
#   CONV3 = auto()
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

    @classmethod
    def arity(cls):
        raise NotImplementedError

    @classmethod
    def make_random(cls):
        raise NotImplementedError

class ConvNode(Node):
    def __init__(self, output_feature_depth, kernel_size):
        super().__init__()
        self.output_feature_depth = output_feature_depth
        self.kernel_size = kernel_size

    def to_block(self, input_feature_depth):
        return ConvBlock(input_feature_depth, self.output_feature_depth, self.kernel_size)

    @classmethod
    def arity(cls):
        return 1

    @classmethod
    def make_random(cls):
        kernel_size = choice([1, 3])
        feature_depth = choice([32, 64, 128, 256])
        return cls(feature_depth, kernel_size)

class ConvBlock(nn.Module):
    def __init__(self, input_feature_depth, output_feature_depth, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        conv_layer = nn.Conv2d(input_feature_depth, output_feature_depth, kernel_size, padding=padding)
        relu_layer = nn.ReLU()
        batch_norm_layer = nn.BatchNorm2d(output_feature_depth)
        self.net = nn.Sequential(conv_layer, relu_layer, batch_norm_layer)

    def forward(self, input):
        return self.net(input)

class Gene(AutoRepr):
    def __init__(self, node, predecessor_indices):
        assert(node.arity() == len(predecessor_indices))
        self.node = node
        self.predecessor_indices = predecessor_indices

class Genome(AutoRepr):
    def __init__(self, input_feature_depth, genes):
        self.input_feature_depth = input_feature_depth
        self.genes = genes

    @classmethod
    def make_random(cls, input_feature_depth, min_length, max_length):
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

        return cls(input_feature_depth, genes)

    @classmethod
    def __instantiable_classes(cls):
        return [ConvNode]
