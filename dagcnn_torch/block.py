from torch import nn

class Block(nn.Module):
    def __init__(self, input_indices):
        super().__init__()
        self.input_indices = input_indices
        self._net = lambda: None

    def forward(self, input):
        return self._net(input)
