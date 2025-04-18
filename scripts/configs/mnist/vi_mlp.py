import sys
sys.path.append("../")
from src.net import MuVarNet
from .setup import dataset_config, prior_config

# Model Config
algo_config = {
    'algorithm' : 'vi',
    'encoder' : MuVarNet(dataset_config['y_dim'],
                          prior_config['x_dim'],),
    'theta_optimizer' : 'adam',
    'q_optimizer' : 'adam',
    'theta_step_size' : 1e-4,
    'q_step_size' : 1e-4,
}
