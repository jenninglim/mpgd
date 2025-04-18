import torch
import math
from src.algorithms.algorithms import Algorithm
from torch.distributions import MultivariateNormal as MVN, Normal

OPTIMIZERS = {'sgd': torch.optim.SGD,
              'rmsprop': torch.optim.RMSprop,
              'adam': torch.optim.Adam}


class VI(Algorithm):
    """
    Implementation of Variational Inference algorithm in VAE.
    (see https://arxiv.org/pdf/1312.6114.pdf).
    """
    def __init__(self,
                 model,
                 encoder,
                 optimizer='sgd',
                 theta_step_size: float = 1e-3,
                 q_step_size: float = 1e-3,
                 device: str = 'cpu',
                 **kwargs):
        super().__init__(model,
                         theta_step_size=theta_step_size,
                         q_step_size=q_step_size,
                         device=device)
        self._encoder = encoder.to(self.device)
        self._decoder_opt = OPTIMIZERS[optimizer](self.model.parameters(),
                                                  lr=theta_step_size)
        self._encoder_opt = OPTIMIZERS[optimizer](self._encoder.parameters(),
                                                  lr=q_step_size)
        self.name = 'VI'

    def _loss(self,
              batch,
              idx,
              return_nll=False,
              n_mc_samples=10):
        mu, var = self._encoder(batch)
        std = var ** 0.5
        dist = Normal(mu, std)
        z = torch.randn(batch.shape[0],
                        n_mc_samples,
                        self.model.latent_size()).to(mu.device) * std.unsqueeze(1) + mu.unsqueeze(1)

        # Compute loss
        nll = - self.model.log_prob(batch, z) / n_mc_samples
        """ Old code should coincide with new
        logz = - 0.5 * torch.log(2 * math.pi * var).sum(-1).unsqueeze(-1)
        logq =  - 0.5 * ((z - mu.unsqueeze(1)) ** 2 / var.unsqueeze(1)).sum(-1)  + logz
        logq = logq.mean(1)
        """
        logq = dist.log_prob(z.permute(1, 0, 2)).sum(-1)
        assert logq.shape == torch.Size([n_mc_samples, batch.shape[0]])
        logq = logq.mean(0).sum(0)  # Mean over the n_mc_samples dimension and sum over batch dimension
        assert nll.shape == logq.shape
        loss = nll + logq
        if return_nll:
            return loss, nll
        return loss

    def _eval_loss(self,
                   batch,
                   idx):
        loss = self._loss(batch, idx)
        return loss.item()

    def step(self,
             batch,
             batch_idx):
        """
        Joint gradient updates of the ELBO
        See Eq 2. https://arxiv.org/pdf/1312.6114.pdf
        """
        # Samples from the posterior
        loss, nll = self._loss(batch, batch_idx, return_nll=True)

        # Update variational distribution and model.
        self._encoder.zero_grad()
        self.model.zero_grad()
        loss.backward()
        self._encoder_opt.step()
        self._decoder_opt.step()
        return nll.item()
