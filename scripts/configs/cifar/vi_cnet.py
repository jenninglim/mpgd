import torch
import sys
sys.path.append("../")
from src.net import MuVarNet
from .setup import dataset_config, prior_config


# Training Algorithm Config
algo_config = {
    'algorithm' : 'vi',
    'encoder' : MuVarNet(dataset_config['y_dim'] * dataset_config['n_channels'],
                         prior_config['x_dim']),
    'theta_step_size' : 1e-4,
    'q_step_size' :1e-4,
    'optimizer': 'rmsprop',
}
