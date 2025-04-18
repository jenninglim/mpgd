import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from torchvision.utils import make_grid
import torchvision.transforms.functional as functional
from pathlib import Path
from torch.func import grad, jvp
from torch import vmap


def anybatchshape(f):
    """
    Wrap the function to allow for different ndim for the batch size.
    """
    def wrapper(self, x,):
        not_batch_shape = x.shape[-1]
        batch_shape = x.shape[:-1]
        # Flatten
        x = x.view(-1, not_batch_shape)
        out = f(self, x)
        return out.view(*batch_shape, -1)
    return wrapper

def myjvp(f, x, v):
    return jvp(f, (x,), (v,))[1]


def divergence(f, x):
    d = x.shape[-1]
    out = 0.
    for i in range(d):
        v = torch.zeros(d, device=x.device)
        v[i] = 1.
        out = out + myjvp(f, x, v)[i]
    return out


def ddlogpdxdt(f, x, score):
    def div_f_t(y):
        return divergence(f, y)
    grad_div_f = vmap(grad(div_f_t), in_dims=0)(x.reshape(-1, x.shape[-1])).reshape(*x.shape)
    grad_f_score = myjvp(f, x, score)
    return - grad_div_f - grad_f_score


def dlogpdt(f, x):
    def div_f_t(y):
        return divergence(f, y)
    div_f = vmap(div_f_t, in_dims=0)(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1])
    return - div_f

def show_images(images,
                show: bool = False,
                path: Optional[Path] = None,
                nrow: int = 1) -> Tuple:
    """Shows and returns figure of image."""
    bad_pixels = torch.logical_and((images <= 1), (images >= 0))
    assert bad_pixels.all(), f"Images should be in [0, 1] found {(bad_pixels.int() - 1).sum().abs()} bad pixels."
    grid = make_grid(images, nrow=nrow)
    grid = functional.to_pil_image(grid)
    fig = plt.imshow(np.asarray(grid))
    plt.axis('off')
    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()
    return fig, grid

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        return data, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def post_ULA(model,
             y,
             n_steps,
             step_size,
             n_particles,
             x0=None):
    if x0 is None:
        x0 = torch.randn(y.shape[0],
                         n_particles,
                         model.latent_size()).to(model.device)
    assert x0.shape == torch.Size([y.shape[0],
                                   n_particles,
                                   model.latent_size()]), f"Expected shape {torch.Size([y.shape[0], n_particles, model.latent_size()])}, got {x0.shape}"
    for _ in range(n_steps):
        with torch.enable_grad():
            x0 = x0.requires_grad_(True)
            log_p_v = model.log_prob(y, x0).sum()
            score = torch.autograd.grad(log_p_v, x0)[0].detach().clone()
            x0 = x0.requires_grad_(False)
        x0 = x0 + step_size * score + (2 * step_size) ** 0.5 * torch.randn_like(x0)
    return x0
