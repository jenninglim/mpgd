from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from src.models import ToyHMM
from src.problems import ToyHMMProblem
from src.algorithms.algorithms import *
from src.algorithms.vi import *
import matplotlib.pyplot as plt

import matplotlib

# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 22
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
NAME = "Test"
torch.random.manual_seed(10)
save_path = Path(f'./results/{NAME}')
save_path.mkdir(exist_ok=True)

## Experiment 1: Toy HMM
# Experiment parameters
n_samples = 1
dim = 100
n_repeats = 1
n_steps = 2000
TRUTH = 100.

# ParEM parameters
STEP_SIZE = 1e-3 / dim/ n_samples
n_particles = 100
q_step_size = 1e-3
t_m = 0.9
q_m = 0.9
device = 'cuda'

# Results record
def empty_record():
    return {'theta': [],
            'loss': []}

results = defaultdict(lambda : defaultdict(list))
baseline_results = defaultdict(list)

generator = ToyHMMProblem(theta=TRUTH, dim=dim)

def choose_eta_theta(gamma_theta,
                     theta_step_size,
                     momentum=0.9):
    gamma_theta = (1 - momentum) / gamma_theta / theta_step_size ** 0.5
    return gamma_theta

b = 2 + dim
K = b + (dim ** 2 - 4) ** 0.5
gamma = 0.29
eta = 2 * K

# Initialize accerated trainer
def init_nesterov_trainer(model, 
                          step_size):
    trainer = AccParticleML_Nesterov(model=model,
                                     theta_step_size=step_size,
                                     q_step_size=q_step_size,
                                     opt_str="sgd",
                                     n_particles=n_particles,
                                     train_size=n_samples,
                                     eval_size=0,
                                     device=device,
                                     gamma_x=gamma,
                                     gamma_theta=gamma,
                                     eta_x=eta,
                                     eta_theta=choose_eta_theta(gamma, step_size),
                                     )
    return trainer

def init_ourtrainer(model, 
                    step_size):
    trainer = AccParticleML_Proposed(model=model,
                                     theta_step_size=step_size ** 0.5,
                                     q_step_size=q_step_size,
                                     opt_str="sgd",
                                     train_size=n_samples,
                                     eval_size=0,
                                     n_particles=n_particles,
                                     gamma_theta=gamma,
                                     eta_theta=choose_eta_theta(gamma, step_size),
                                     gamma_x=gamma,
                                     eta_x=eta,
                                     device=device)
    return trainer

def init_trainer(model,
                 step_size):
    trainer = ParticleML(model=model,
                         theta_step_size=step_size ** 0.5,
                         q_step_size=q_step_size,
                         opt_str="sgd",
                         train_size=n_samples,
                         eval_size=0,
                         n_particles=n_particles,
                         device=device)
    return trainer


configs = {'x': {'step_size': STEP_SIZE},
           'y': {'step_size': STEP_SIZE*5},
           'v': {'step_size': STEP_SIZE*0.5},
           'z': {'step_size': STEP_SIZE*50}}

configs_plot = {'x': {'c': 'red', 'marker': 'o', 'markevery': 150},
                'y': {'c': 'blue', 'marker': '*', 'markevery': 150},
                'v': {'c': 'green', 'marker': 'v', 'markevery': 150},
                'z': {'c': 'brown', 'marker': 'x', 'markevery': 150},}

trainers = {
            'nesterov': init_nesterov_trainer,
            'ours': init_ourtrainer,
            }

trainers_plot = {'nesterov': {'linestyle': '-',
                              'alpha' : 0.8},
                 'ours': {'linestyle': '--',
                          'alpha' : .8}}

# Run experiment
idx = torch.arange(n_samples)
for i in range(n_repeats):
    dataset = generator.sample(n_samples).to(device)
    # Baseline
    model = ToyHMM(dim=dim).to(device)
    for config_name, config in configs.items():
        if config['step_size'] < STEP_SIZE * 10:
            trainer = init_trainer(model, config['step_size'])
            result = empty_record()
            for step in tqdm(range(n_steps)):
                loss = trainer.step(dataset, idx)
                result['theta'].append(trainer.model.theta.item())
                result['loss'].append(loss)
            baseline_results[config_name].append(result)

    # Proposal
    for config_name, config in configs.items():
        for trainer_name, trainer_init in trainers.items():
            model = ToyHMM(dim=dim).to(device)
            trainer = trainer_init(model, **config)
            result = empty_record()
            for step in tqdm(range(n_steps)):
                loss = trainer.step(dataset, idx)
                result['theta'].append(trainer.model.theta.item())
                result['loss'].append(loss)
            results[config_name][trainer_name].append(result)

# Init summary
summary = defaultdict(dict)
baseline_summary = {}

# Calculate the mean of baseline
for baseline_name, baseline_result in baseline_results.items():
    thetas = [result['theta'] for result in baseline_result]
    losss = [result['loss'] for result in baseline_result]
    theta_mu = np.mean(np.stack(thetas, axis=0), axis=0)
    loss_mu = np.mean(np.stack(losss, axis=0), axis=0)
    baseline_summary[baseline_name] = {'theta': theta_mu, 'loss': loss_mu}


# Calculate the mean of proposals
for config, config_results in results.items():
    for trainer_name, trainer_results in config_results.items():
        thetas = np.stack([result['theta'] for result in trainer_results], axis=1)
        losss = np.stack([result['loss'] for result in trainer_results], axis=1)
        plt.plot(range(len(thetas)),
                 thetas,
                 label=trainer_name+'_'+config,
                 **trainers_plot[trainer_name],
                 **configs_plot[config])
        theta_mu = np.mean(thetas, axis=1)
        loss_mu = np.mean(losss, axis=1)
        summary[config][trainer_name] = {'theta': theta_mu, 'loss': loss_mu}
plt.savefig(save_path / f'toyhmm3_all.pdf')
plt.clf()
plt.close()

for qoi in empty_record().keys():
    # Plot Baseline results
    for baseline_name, baseline_results in baseline_summary.items():
        plt.plot(baseline_results[qoi],
                 **configs_plot[baseline_name],
                 label=baseline_name,
                 alpha=0.5,
                 c='black')
    
    # Plot Truth
    if qoi == 'theta':
        plt.axhline(y=TRUTH, color='black', linestyle=':')

    # Plot proposals
    for config, config_results in summary.items():
        for trainer_name, trainer_results in config_results.items():
            plt.plot(trainer_results[qoi],
                     label=trainer_name+'_'+config,
                     **trainers_plot[trainer_name],
                     **configs_plot[config])
        plt.legend()
        if qoi == 'theta':
            plt.ylim(0, 110)
        plt.savefig(save_path / f'toyhmm3_{qoi}.pdf')
    plt.clf()
    plt.close()