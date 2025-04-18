import sys
from collections import defaultdict
import numpy as np
import torch
from src.models import ToyHMM
from src.problems import ToyHMMProblem
from src.algorithms.algorithms import *
from tqdm import tqdm
from src.algorithms.other_discretizations import *
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

NAME = 'role_of_gc'

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

## Experiment 1: Toy HMM
# Experiment parameters
n_samples = 100
n_repeats = 1
n_steps = 2000

# ParEM parameters
n_particles = 100
device = 'cuda'
gamma = 0.293
gamma_theta = gamma
b = 2 + n_samples
K = b + (n_samples ** 2 - 4) ** 0.5
gamma_x = gamma
eta_x = 2 * K
eta_theta = 2 * K
Q_STEP_SIZE = 2.18e-3
T_STEP_SIZE = 6.68e-3
TRUTH = 100

# Results record
def empty_record():
    return {'theta': [],
            'loss': []}

results = defaultdict(lambda : defaultdict(list))
baseline_results = defaultdict(list)

generator = ToyHMMProblem(theta=TRUTH)


def init_nc_trainer(model, 
                    factor):
    trainer = MPD_NC(model=model,
                     theta_step_size=T_STEP_SIZE * factor,
                     q_step_size=Q_STEP_SIZE * factor,
                     train_size=n_samples,
                     eval_size=0,
                     n_particles=n_particles,
                     gamma_theta=gamma_theta,
                     eta_theta=eta_theta,
                     gamma_x=gamma_x,
                     eta_x=eta_x,
                     device=device)
    return trainer


def init_nct_trainer(model, 
                     factor):
    trainer = MPD_NCTwo(model=model,
                        theta_step_size=T_STEP_SIZE * factor,
                        q_step_size=Q_STEP_SIZE * factor,
                        train_size=n_samples,
                        eval_size=0,
                        n_particles=n_particles,
                        gamma_theta=gamma_theta,
                        eta_theta=eta_theta,
                        gamma_x=gamma_x,
                        eta_x=eta_x,
                        device=device)
    return trainer


# Initialize Proposed trainer
def init_c_trainer(model, 
                   factor):
    trainer = MPD_Proposed(model=model,
                           theta_step_size=T_STEP_SIZE * factor,
                           q_step_size=Q_STEP_SIZE * factor,
                           train_size=n_samples,
                           eval_size=0,
                           n_particles=n_particles,
                           gamma_theta=gamma_theta,
                           eta_theta=eta_theta,
                           gamma_x=gamma_x,
                           eta_x=eta_x,
                           device=device)
    return trainer


configs = {'1': {'factor': 1},
           '1.1': {'factor': 1.01},
           '0.9': {'factor': 0.99},}

configs_plot = {'1': {'linestyle': '-'},
                '1.1': {'linestyle': '-'},
                '0.9': {'linestyle': '-'}}

trainers = {
            'NC': init_nc_trainer,
            'NC2': init_nct_trainer,
            'Ours': init_c_trainer,
            }

trainers_plot = {'NC2': {'c': 'green',
                         'alpha' : 0.5},
                'NC': {'c': 'red',
                       'alpha' : 0.5},
                 'Ours': {'c': 'blue',
                          'alpha' : 0.5}}

# Run experiment
idx = torch.arange(n_samples)
for i in range(n_repeats):
    dataset = generator.sample(n_samples).to(device)
    # Proposal
    for config_name, config in configs.items():
        for trainer_name, trainer_init in trainers.items():
            model = ToyHMM(dim=1).to(device)
            trainer = trainer_init(model, **config)
            result = empty_record()
            for step in tqdm(range(int(n_steps))):
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
        linspace = np.arange(len(thetas)) * configs[config]['factor']
        plt.plot(linspace,
                 thetas,
                 label=trainer_name+'_'+config,
                 **trainers_plot[trainer_name],
                 **configs_plot[config])
        theta_mu = np.mean(thetas, axis=1)
        loss_mu = np.mean(losss, axis=1)
        summary[config][trainer_name] = {'theta': theta_mu, 'loss': loss_mu}

save_dir = Path(f'./results/{NAME}')
save_dir.mkdir(exist_ok=True, parents=True)
plt.savefig(save_dir / 'all.pdf')
plt.clf()
plt.close()

for qoi in empty_record().keys():
    # Plot Truth
    # Plot proposals
    for i, (config, config_results) in enumerate(summary.items()):
        for trainer_name, trainer_results in config_results.items():
            if qoi == 'theta':
                plt.axhline(y=TRUTH, color='black', linestyle=':')

            linspace = np.arange(len(trainer_results[qoi])) * configs[config]['factor']
            plt.figure(1)
            plt.plot(linspace,
                     trainer_results[qoi],
                     label=trainer_name+'_'+config,
                     **trainers_plot[trainer_name],
                     **configs_plot[config])
            plt.xlim(0, n_steps)

            plt.figure(i+2)
            plt.plot(linspace,
                     trainer_results[qoi],
                     label=trainer_name+'_'+config,
                     **trainers_plot[trainer_name],
                     **configs_plot[config])
            plt.xlim(0, n_steps)

    for i, config_name in enumerate(summary.keys()):
        plt.figure(i+2)
        plt.legend()
        plt.savefig(save_dir / f'{config_name}_{qoi}.pdf')
        plt.clf()
        plt.close()
    plt.figure(1)
    plt.legend()
    plt.savefig(save_dir / f'{qoi}.pdf')
    plt.clf()
    plt.close()
