import random, os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from operator import itemgetter
from pathlib import Path
from src.utils import post_ULA
from pytorch_fid.inception import InceptionV3
from ignite.metrics.gan import FID
import torch.nn as nn
import PIL.Image as Image


def set_seed(seed):
    '''
    Code snippet taken from https://alexandruburlacu.github.io/posts/2022-11-22-mlops-fable.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seeded_dl(seed):
    '''
    Code snippet taken from https://alexandruburlacu.github.io/posts/2022-11-22-mlops-fable.
    '''
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed + 10) 
    dl = lambda dataset, batch_size, num_workers=1: DataLoader(dataset,
                                                               batch_size=batch_size,
                                                               num_workers=num_workers, 
                                                               shuffle=True,
                                                               worker_init_fn=seed_worker,
                                                               generator=g)
    return dl


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299),
                                     Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    arr = torch.stack(arr)
    if arr.shape[1] == 1:
        arr = arr.repeat(1,3,1,1)
    return arr

def reconstruct(imgs, model, q_step_size, n_particles, n_steps):
    device = model.device
    with torch.no_grad():
        x0 = model.prior.sample(n_particles * imgs.shape[0]).to(device).reshape(imgs.shape[0],
                                                                                n_particles,
                                                                                -1)
        particles = post_ULA(model,
                            imgs.to(device),
                            n_steps,
                            q_step_size,
                            n_particles,
                            x0=x0.to(device))
    imgs = model.decoder(particles).to(imgs.device)
    return imgs


def get_best_model_path(config_name, seed):
    path = Path(f"checkpoint/{config_name}")
    split_checkpoints = [str(pt)[:-3].split("_") for pt in path.glob("*.pt")]
    split_checkpoints = list(filter(lambda x: (len(x) == 7) and (x[3] == str(seed)), split_checkpoints))
    assert len(split_checkpoints) > 0
    ind, best_epoch = max(enumerate([int(split[-2]) for split in split_checkpoints]),
                          key=itemgetter(1))
    loss = split_checkpoints[ind][-1]
    best_model_path = Path(f"checkpoint/{config_name}/best_seed_{seed}_model_{best_epoch}_{loss}.pt")
    assert best_model_path.exists()
    return best_model_path


def pytorch_fid(dims=2048, device='cuda'):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    class WrapperInceptionV3(nn.Module):

        def __init__(self, fid_incv3):
            super().__init__()
            self.fid_incv3 = fid_incv3

        @torch.no_grad()
        def forward(self, x):
            y = self.fid_incv3(x)
            y = y[0]
            y = y[:, :, 0, 0]
            return y

    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.eval()

    m = FID(num_features=dims, feature_extractor=wrapper_model, device=device)
    return m