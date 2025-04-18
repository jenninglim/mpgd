import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from src.utils import anybatchshape

class ThirtyTwoConvT(nn.Module):
    def __init__(self,
                 in_dim,
                 ngf=256,
                 transform=None,
                 use_bn=False):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_dim,
                                        ngf * 8,
                                        kernel_size=8,
                                        stride=1,
                                        bias=False)
        self.conv2 = nn.ConvTranspose2d(ngf * 8,
                                        ngf * 4,
                                        padding=1,
                                        kernel_size=4,
                                        stride=2,
                                        bias=False)
        self.conv3 = nn.ConvTranspose2d(ngf * 4,
                                        ngf * 2,
                                        padding=1,
                                        kernel_size=4,
                                        stride=2,
                                        bias=False)
        self.conv4 = nn.ConvTranspose2d(ngf * 2,
                                        3,
                                        padding=1,
                                        kernel_size=3,
                                        stride=1,
                                        bias=False)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(ngf * 8)
            self.bn2 = nn.BatchNorm2d(ngf * 4)
            self.bn3 = nn.BatchNorm2d(ngf * 2)

        self.transform = transform
    
    @anybatchshape
    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.bn1(x) if self.use_bn else x

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.bn2(x) if self.use_bn else x

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.bn3(x) if self.use_bn else x

        x = self.conv4(x)
        if self.transform is not None:
            x = self.transform(x)
        return x


class MLP(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_hidden=512,
                 transform=None,
                 use_bn=False,
                 use_weight_norm=False,
                 act='leaky_relu'):
        super().__init__()
        if act == 'leaky_relu':
            act = nn.LeakyReLU
        elif act == 'relu':
            act = nn.ReLU
        self.fc1 = nn.Linear(in_dim, n_hidden)
        self.fc1 = nn.utils.weight_norm(self.fc1) if use_weight_norm else self.fc1
        self.act1 = act()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.utils.weight_norm(self.fc2) if use_weight_norm else self.fc2
        self.act2 = act()
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.utils.weight_norm(self.fc3) if use_weight_norm else self.fc3
        self.act3 = act()
        self.fc4 = nn.Linear(n_hidden, out_dim)
        self.out_dim = out_dim
        self.transform = transform
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.LayerNorm(n_hidden)
            self.bn2 = nn.LayerNorm(n_hidden)
            self.bn3 = nn.LayerNorm(n_hidden)


    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.bn1(out) if self.use_bn else out

        out = self.fc2(out)
        out = self.act2(out)
        out = self.bn2(out) if self.use_bn else out

        out = self.fc3(out)
        out = self.act3(out)
        out = self.bn3(out) if self.use_bn else out

        out = self.fc4(out)
        if not self.transform is None:
            out = self.transform(out)
        return out


class ConvNet(nn.Module):
    def __init__(self,
                 x_dim: int,
                 n_in_channel: int = 1,
                 n_out_channel: int = 16,
                 n_hidden: int = 512):
        """
        :param x_dim: Dimension of the latent variable.
        :param n_in_channel: number of channels of the images.
        :param n_out_channel: number of channel output of the conv layer.
        :param n_hidden: dimension of the hidden (linear) layer.
        """
        super().__init__()
        self.x_dim = x_dim
        self.conv1 = nn.Conv2d(n_in_channel,
                               n_out_channel,
                               kernel_size=3,
                               stride=1,
                               padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_out_channel,
                               n_out_channel * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 8 * n_out_channel * 2, n_hidden)
        self.fc2 = nn.Linear(n_hidden, x_dim)

    def forward(self, y):
        y = self.conv1(y)
        y = F.leaky_relu(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = F.leaky_relu(y)
        y = self.pool2(y)
        y = y.flatten(start_dim=1)
        y = self.fc1(y)
        y = F.leaky_relu(y)
        y = self.fc2(y)
        return y


class MuVarNet(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_hidden=512,
                 use_weight_norm=False,
                 use_bn=False,
                 act='leaky_relu'):
        super().__init__()
        self.out_dim = out_dim
        self.net = MLP(in_dim,
                       out_dim * 2,
                       n_hidden=n_hidden,
                       use_weight_norm=use_weight_norm,
                       act=act,
                       use_bn=use_bn)
    
    def forward(self, x):
        out_dim = self.out_dim
        out = self.net(x)
        mu = out[..., :out_dim]
        var =  F.softplus(out[..., out_dim:])
        return mu, var + 1e-5


class MuCovNet(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_hidden=256):
        super().__init__()
        self.out_dim = out_dim
        self.net = MLP(in_dim,
                       out_dim  + out_dim ** 2,
                       n_hidden=n_hidden)
    
    def forward(self, x):
        out_dim = self.out_dim
        out = self.net(x)
        mu = out[..., :out_dim]
        L = out[..., out_dim:].view(*out.shape[:-1], out_dim, out_dim)
        log_cov = (L + torch.transpose(L, -2, -1)) / 2. / out_dim
        return mu, log_cov
