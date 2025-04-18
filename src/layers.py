import torch
import torch.nn as nn
import torch.nn.functional as F


class Deterministic(nn.Module):
    """
    The Deterministic Layer used in NLVM.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation=F.leaky_relu):
        super(Deterministic, self).__init__()

        self.activation = activation

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1,
                              padding=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1,
                               padding=1)


    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = out + x  # Skip connection
        return out


class Projection(nn.Module):
    """
    The Projection Layer used in NLVM.
    """
    def __init__(self,
                 in_dim: int,
                 ngf: int = 16,
                 coef: int = 4,
                 activation=F.leaky_relu):
        super(Projection, self).__init__()

        self.activation = activation
        self.ngf = 16
        self.coef = 4

        self.linear = nn.Linear(in_dim, coef * ngf * ngf)
        self.deconv1 = nn.ConvTranspose2d(coef, ngf * coef, kernel_size=5,
                                          stride=1, padding=2, bias=False)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = out.view(out.size(0), self.coef, self.ngf, self.ngf).contiguous()
        out = self.deconv1(out)
        out = self.activation(out)
        return out


class Output(nn.Module):
    """
    The Output Layer used in NLVM.
    """
    def __init__(self,
                 x_in: int,
                 nc: int):
        super(Output, self).__init__()
        self.output_layer = nn.ConvTranspose2d(x_in, nc, kernel_size=4,
                                               stride=2, padding=1)

    def forward(self, x):
        out = self.output_layer(x)
        return out.flatten(start_dim=-3)
