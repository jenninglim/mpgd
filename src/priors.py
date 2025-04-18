import torch
from abc import ABCMeta, abstractmethod
from .net import MuVarNet, MuCovNet
import torch.nn as nn
from .utils import *
import numpy as np
from math import ceil
import math
from torch.func import grad
from src.utils import anybatchshape


class Prior(nn.Module):
    def __init__(self, x_dim, device='cpu'):
        super().__init__()
        self.x_dim = x_dim
        self.device = device
        self.pretrain = False

    def forward(self, num_samples=1):
        samples = self.sample(num_samples)
        logp = self.log_prob(samples)
        return samples, logp

    def log_prob(self, x):
        if self.pretrain:
            return self.normal_logp(x)
        else:
            return self._log_prob(x)

    @abstractmethod
    def _log_prob(self, x):
        raise NotImplementedError()

    def sample(self, n_samples):
        if self.pretrain:
            return torch.randn(n_samples,
                               self.x_dim).to(self.device)
        else:
            return self._sample(n_samples)

    @abstractmethod
    def _sample(self, n_samples):
        raise NotImplementedError()
        
    def score(self, x):
        """
        Compute the gradient of the log p(x) w.r.t. x.
        """
        g_lprior = grad(self.log_prob)
        return g_lprior(x)

    def normal_logp(self, x):
        return - (x ** 2).sum(-1) / 2.

    def set_pretrain(self,):
        for param in self.parameters():
            param.requires_grad = False
        self.pretrain = True 

    def set_nopretrain(self,):
        for param in self.parameters():
            param.requires_grad = True
        self.pretrain = False

    @anybatchshape
    def vscore(self, x):
        vscore_f = vmap(self.score, in_dims=0)
        return vscore_f(x)

class VampPrior(Prior):
    def __init__(self,
                 x_dim,
                 n_pseudo,
                 pseudo_latent_dim,
                 device='cpu',
                 train_pseudo=True,
                 **kwargs):
        super().__init__(x_dim, device=device)
        pseudo_points = torch.randn(n_pseudo, pseudo_latent_dim, device=device) * 10
        self.n_pseudo = n_pseudo
        if train_pseudo:
            self.pseudo_points = nn.Parameter(pseudo_points).to(device)


class VAMPVarPrior(VampPrior):
    def __init__(self,
                 x_dim,
                 n_pseudo,
                 pseudo_latent_dim,
                 device='cpu',
                 n_hidden=512,
                 train_pseudo=True,
                 use_weight_norm=False,
                 use_bn=False,
                 **kwargs):
        super().__init__(x_dim,
                         n_pseudo,
                         pseudo_latent_dim,
                         device=device,
                         train_pseudo=train_pseudo)
        self.encoder = MuVarNet(pseudo_latent_dim,
                                x_dim,
                                n_hidden=n_hidden,
                                use_weight_norm=use_weight_norm,
                                use_bn=use_bn).to(device)


    def _log_prob(self, x):
        mus, sigs = self.encoder(self.pseudo_points)
        mse = - ((x.unsqueeze(-2) - mus) ** 2 / 2 / sigs).sum(-1)
        logz = 0.5 * torch.log(sigs).sum(-1) + 0.5 * np.log(2 * math.pi) * self.x_dim
        comps = mse - logz
        log_prob = (comps).logsumexp(-1) - np.log(self.n_pseudo)
        return log_prob

    def _sample(self, n_samples):
        mus, sigs = self.encoder(self.pseudo_points)
        randint = np.random.randint(self.n_pseudo, size=n_samples)
        samples = torch.randn(n_samples, self.x_dim, device=self.device)
        samples = samples * sigs[randint] ** 0.5
        samples = samples + mus[randint]
        return samples


class VAMPCovPrior(Prior):
    def __init__(self,
                 x_dim,
                 n_pseudo,
                 pseudo_latent_dim,
                 device='cpu',
                 train_pseudo=True,
                 **kwargs):
        super().__init__(x_dim,
                         n_pseudo,
                         pseudo_latent_dim,
                         device=device,
                         train_pseudo=train_pseudo)
        self.encoder = MuVarNet(pseudo_latent_dim, x_dim).to(device)
        
    def _log_prob(self, x):
        mus, log_cov = self.encoder(self.pseudo_points)
        inv_cov = torch.matrix_exp(- log_cov)
        malanobis = - 0.5 * torch.einsum("...i,...ij, ...j->...",
                                   (x - mus), inv_cov, (x - mus)) 
        logz = 0.5 * torch.einsum("...ii->...", log_cov) + 0.5 * np.log(2 * math.pi) * self.x_dim
        comps = malanobis - logz
        log_prob = (comps).logsumexp(-1) - np.log(self.n_pseudo)
        return log_prob

    def _sample(self, n_samples):
        with torch.no_grad():
            mus, log_cov = self.encoder(self.pseudo_points)
            det = torch.einsum("...ii->...", log_cov)
            cov = torch.matrix_exp(log_cov) + torch.eye(log_cov.shape[-1], device=self.device) * 1e-8
            L = torch.cholesky(cov)
            randint = np.random.randint(self.n_pseudo, size=n_samples)
            samples = torch.randn(n_samples, self.x_dim, device=self.device)
            samples = torch.einsum("...ij,...j->...i", L[randint], samples)
            samples = samples + mus[randint]
        return samples


class FlowPrior(Prior):
    def __init__(self,
                 x_dim,
                 network_type,
                 n_hidden=512,
                 n_layers=48,
                 device='cpu',
                 use_bn=False,
                 use_weight_norm=False,
                 use_vamp=True,
                 n_pseudo=50,
                 **kwargs):
        super().__init__(x_dim, device=device)

        if use_vamp:
            self.base = VAMPVarPrior(x_dim,
                                    device=device,
                                    use_weight_norm=use_weight_norm,
                                    n_hidden=n_hidden,
                                    use_bn=use_bn,
                                    n_pseudo=n_pseudo,
                                    **kwargs)
        else:
            self.base = GaussianMixture(
                n_pseudo,
                x_dim,
                loc=[np.random.randn(x_dim) for i in range(n_pseudo)] ,
                trainable=True,
            )

        flows = []
        dim_in = ceil(x_dim / 2)
        dim_out = x_dim if x_dim % 2 == 0 else x_dim - 1
        flows = []
        for i in range(n_layers):
            param_map = nf.nets.MLP([dim_in, *network_type, dim_out], init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(x_dim, mode='swap'))
            flows.append(nf.flows.ActNorm(x_dim))

        self.model = nf.NormalizingFlow(self.base, flows).to(device)
    
    def vscore(self, x):
        with torch.enable_grad():
            x.requires_grad = True
            log_p = self.log_prob(x).sum()
            score = torch.autograd.grad(log_p, x)[0]
        return score

    def _sample(self, n_samples, max_retries=50):
        samples = self.model.sample(n_samples)[0]
        retries = 0
        while samples.isnan().any() and retries < max_retries:
            nan_idx = samples.isnan().sum(-1) > 0
            replacements = self.model.sample(int(nan_idx.sum()))[0]
            samples[nan_idx] = replacements
        return samples
    
    def _log_prob(self, x):
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, self.x_dim)
        out = self.model.log_prob(x)
        out = out.reshape(batch_shape)
        return out