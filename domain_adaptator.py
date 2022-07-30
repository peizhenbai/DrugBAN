import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):
    """The gradient reversal layer (GRL)

    This is defined in the DANN paper http://jmlr.org/papers/volume17/15-239/15-239.pdf

    Forward pass: identity transformation.
    Backward propagation: flip the sign of the gradient.

    From https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/layers.py
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_size=128, n_class=1, bigger_discrim=True):

        super(Discriminator, self).__init__()
        output_size = 256 if bigger_discrim else 128

        self.bigger_discrim = bigger_discrim
        self.fc1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size) if bigger_discrim else nn.Linear(output_size, n_class)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(output_size, n_class)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        if self.bigger_discrim:
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        return x