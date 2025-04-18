from abc import abstractmethod
import torch
from torch.func import functional_call
import torch.nn as nn
from torch import Size, vmap
from torch.func import grad
from .net import MLP, ThirtyTwoConvT
from math import pi
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal as MVN, Normal, Independent


class Model(nn.Module):
    @abstractmethod
    def log_prior(self, x):
        raise NotImplementedError("log_prior not implemented.")

    @abstractmethod
    def log_likeli(self, y, x):
        '''
        y: Shape([batch_size, y_dim])
        x: Shape([batch_size, x_dim])
        '''
        raise NotImplementedError("log_likeli not implemented.")

    @abstractmethod
    def sample(self, n_samples):
        raise NotImplementedError("sample not implemented.")

    def forward(self, y, x):
        return self.log_likeli(y, x) + self.log_prior(x)
    
    def log_prob(self, y, x):
        '''
        y: Shape([batch_size, y_dim])
        x: Shape([batch_size, n_particles, x_dim])
        '''
        vlogp = vmap(self.forward, in_dims=(None, 0))
        vvlogp = vmap(vlogp, in_dims=(0, 0))
        out = vvlogp(y, x)
        return out.sum(1).sum(0)

    def ell(self, parameters, y, x):
        return functional_call(self, parameters, (y, x))

    def latent_size(self):
        if hasattr(self, "latent_dim"):
            return self.latent_dim
        else:
            raise Exception("Missing latent dim attribute.")

    def score_likeli(self, y, x):
        """
        Compute gradient of log p(y|x) w.r.t. x.
        """
        g_llikeli = grad(self.log_likeli, argnums=1)
        return g_llikeli(y, x)

    def score_prior(self, x):
        """
        Compute the gradient of the log p(x) w.r.t. x.
        """
        score = grad(self.log_prior, argnums=0)
        return score(x)

    def score_log_p(self, y, x):
        vmap_gloglikeli = vmap(vmap(self.score_likeli,
                                    in_dims=(None, 0)),
                                in_dims=(0, 0))
        gloglikeli = vmap_gloglikeli(y, x)
        vmap_score_prior = vmap(vmap(self.score_prior,
                                    in_dims=0),
                                in_dims=0)
        particle_prior_score = vmap_score_prior(x)
        assert particle_prior_score.shape == x.shape
        assert gloglikeli.shape == x.shape
        return gloglikeli + particle_prior_score

class ToyHMM(Model):
    def __init__(self, dim, scale=1.):
        super().__init__()
        centre = torch.zeros(1)
        self.theta = torch.nn.Parameter(centre)
        self.latent_dim = dim
        self.scale = scale

    def log_prior(self, x):
        dim = self.latent_dim
        var = self.scale ** 2
        return - ((x - self.theta) ** 2).sum(-1) / 2. / var - dim / 2 * np.log(2 * pi * var)

    def log_likeli(self, y, x):
        dim = self.latent_dim
        return - ((y - x) ** 2).sum(-1) / 2  - dim / 2 * np.log(2 * pi)

class ScaleToyHMM(Model):
    def __init__(self, dim, scale=1.):
        super().__init__()
        centre = torch.zeros(1)
        scale = torch.ones(1) * scale
        self.theta = torch.nn.Parameter(centre)
        self.scale = torch.nn.Parameter(scale)
        self.latent_dim = dim

    def log_prior(self, x):
        dim = self.latent_dim
        var = self.scale ** 2
        return - ((x - self.theta) ** 2).sum(-1) / 2. / var - dim / 2 * torch.log(2 * pi * var)

    def log_likeli(self, y, x):
        dim = self.latent_dim
        return - ((y - x) ** 2).sum(-1) / 2. - dim / 2 * np.log(2 * pi)

def log_normal_prob(x, sigma2=1., mu=0.):
    sigma2 = sigma2
    return - ((x - mu) ** 2).sum(-1) / (2 * sigma2)\
          - x.shape[0] / 2 * torch.log(2 * pi * sigma2)

class BNN(Model):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_hidden=40):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden

        self.n_weights1 = self.in_dim * self.n_hidden
        self.n_weights2 = self.n_hidden * self.out_dim
        self.n_weights = self.n_weights1 + self.n_weights2
        self.latent_dim = self.n_weights 
        
        self.alpha_mu = torch.nn.Parameter(torch.randn(1) * 0.1)
        self.beta_mu = torch.nn.Parameter(torch.randn(1) * 0.1)

        self.alpha = torch.nn.Parameter(torch.tensor(0.))
        self.beta = torch.nn.Parameter(torch.tensor(0.))
    
    def vec_to_params(self, x):
        assert x.ndim == 1
        assert x.shape[-1] == self.n_weights
        w0 = x[:self.in_dim * self.n_hidden].reshape(self.n_hidden, self.in_dim)
        w1 = x[self.in_dim * self.n_hidden:].reshape(self.out_dim, self.n_hidden)
        return w0, w1
    
    def evaluate(self, x, z):
        w0, w1 = self.vec_to_params(x)
        fc1 = torch.einsum("ij,j->i", w0, z) 
        act1 = torch.tanh(fc1)
        fc2 = torch.einsum("ij,j->i", w1, act1)
        return fc2
    
    def logits(self, y, x):
        (z, f) = y
        out = self.evaluate(x, z)
        return F.log_softmax(out, dim=-1)
    
    def log_likeli(self, y, x):
        (z, f) = y
        logits = self.logits(y, x)
        out = logits[f[None]][0]
        return out
    
    def log_prior(self, x):
        w0, w1 = self.vec_to_params(x)
        w0 = w0.reshape(self.n_hidden * self.in_dim)
        w1 = w1.reshape(self.n_hidden * self.out_dim)
        w0_prob = log_normal_prob(w0, F.softplus(self.alpha), self.alpha_mu)
        w1_prob = log_normal_prob(w1, F.softplus(self.beta), self.beta_mu)
        return w0_prob + w1_prob 
    
    def log_prob(self, y, x):
        vloglikeli = vmap(vmap(self.log_likeli, in_dims=(None, 0)),
                          in_dims=(0, None))
        vlog_prior = vmap(self.log_prior, in_dims=0)
        logprior = vlog_prior(x)
        loglikeli = vloglikeli(y, x)
        assert logprior.shape == torch.Size([x.shape[0]])
        assert loglikeli.shape == torch.Size([y[0].shape[0], x.shape[0]])
        out = logprior + loglikeli.sum(0)
        return out.sum()


class VAE(Model):
    def __init__(self,
                 y_dim,
                 x_dim,
                 likelihood_noise,
                 prior,
                 device='cpu',
                 transform=None,
                 net='mlp',
                 n_channels=1,
                 **kwargs):
        super().__init__()
        self.latent_dim = x_dim
        self.y_dim = y_dim
        if net == 'mlp':
            self.decoder = MLP(x_dim,
                               y_dim * n_channels,
                               transform=transform).to(device)
        elif net == 'cnet':
            self.decoder = ThirtyTwoConvT(x_dim,
                                        transform=transform).to(device)
        self.device = device
        self.likelihood_noise = likelihood_noise
        self.prior = prior

    def log_likeli(self, y, x):
        var = self.likelihood_noise
        y_prime = self.decoder(x)
        d = y.shape[-1]
        '''
        dist = MVN(loc=y_prime,
                   scale_tril=torch.eye(d, device=self.device) * var ** 0.5)
        out = dist.log_prob(y)
        '''
        dist = Normal(y_prime, var ** 0.5)
        out = dist.log_prob(y).sum(-1)
        assert out.shape == torch.Size([x.shape[0], x.shape[1]])
        assert not y_prime.isnan().any(), "NaN in y_prime"
        return out

    def log_prior(self, x):
        return self.prior.log_prob(x)

    def sample(self, n_samples, max_retries=50):
        latent = self.prior.sample(n_samples)
        nan_idx = latent.isnan().sum(-1) > 0
        retries = 0
        while nan_idx.sum() > 0 and retries < max_retries:
            print(f"Warning: nan in latent found {nan_idx.sum()}.")
            latent[nan_idx] = self.prior.sample(int(nan_idx.sum().cpu()))
            nan_idx = latent.isnan().sum(-1) > 0
            retries += 1
        assert nan_idx.sum() == 0
        y = self.decoder(latent)
        nan_idx = y.isnan().sum(-1) > 0
        retries = 0
        while nan_idx.sum() > 0 and retries < max_retries:
            print(f"Warning: nan in y found {nan_idx.sum()}.")
            latent[nan_idx] = self.prior.sample(int(nan_idx.sum().cpu()))
            y[nan_idx] = self.decoder(latent[nan_idx])
            nan_idx = y.isnan().sum(-1) > 0
            retries += 1
        return y, latent

    def log_prob(self, y, x):
        '''
        A non-vmap approach to log_prob.
        This is to enable batch norm for instance.
        '''
        batch_size = y.shape[0]
        n_particles = x.shape[1]
        assert x.shape[0] == y.shape[0] 
        log_likeli = self.log_likeli(y.unsqueeze(1), x) 
        assert not log_likeli.isnan().any(), "NaN in log_likeli"
        assert log_likeli.shape == Size([batch_size, n_particles])
        log_prior =  self.log_prior(x)
        assert log_likeli.shape == log_prior.shape
        assert not log_prior.isnan().any(), "NaN in log_prior"
        out = log_likeli + log_prior
        assert out.shape == Size([batch_size, n_particles])
        return out.sum(1).sum(0)

    def score_log_p(self, y, x):
        vmap_gloglikeli = vmap(vmap(self.score_likeli,
                                    in_dims=(None, 0)),
                                in_dims=(0, 0))
        gloglikeli = vmap_gloglikeli(y, x)
        with torch.enable_grad():
            x.requires_grad_(True)
            log_prior_prob = self.prior.log_prob(x)
            particle_prior_score= torch.autograd.grad(log_prior_prob.sum(), x)[0]
            x.requires_grad_(False)
        assert particle_prior_score.shape == x.shape
        assert gloglikeli.shape == x.shape
        return gloglikeli + particle_prior_score