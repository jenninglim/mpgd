import sys
from collections import defaultdict
import numpy as np
import torch
from src.models import ToyHMM
from src.problems import ToyHMMProblem
from src.algorithms.algorithms import *
from src.algorithms.vi import *
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path


NAME = 'different_regimes'
save_dir = Path('./results') / NAME
save_dir.mkdir(parents=True, exist_ok=True)

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
STEP_SIZE = 1e-2
theta_step_size = STEP_SIZE / n_samples
q_step_size = STEP_SIZE
device = 'cuda'
TRUTH = 100.

# Results record
def empty_record():
    return {'theta': [],
            'loss': []}

results = defaultdict(lambda : defaultdict(list))
baseline_results = defaultdict(list)

generator = ToyHMMProblem(theta=TRUTH)

def choose_eta_theta(gamma_theta,
                     momentum=0.9,
                     theta_step_size=theta_step_size,):
    gamma_theta = (1 - momentum) / gamma_theta / theta_step_size
    return gamma_theta

# Initialize accerated trainer
def init_atrainer(model, 
                  gamma_theta,
                  eta_theta,
                  gamma_x,
                  eta_x):
    trainer = MPD_Proposed(model=model,
                           theta_step_size=theta_step_size,
                           q_step_size=q_step_size,
                           train_size=n_samples,
                           eval_size=0,
                           n_particles=n_particles,
                           gamma_theta=gamma_theta,
                           eta_theta=eta_theta,
                           gamma_x=gamma_x,
                           eta_x=eta_x,
                           device=device)
    return trainer

def init_trainer(model):
    trainer = ParticleML(model=model,
                         theta_step_size=theta_step_size,
                         q_step_size=q_step_size,
                         train_size=n_samples,
                         eval_size=0,
                         n_particles=n_particles,
                         device=device)
    return trainer

K = 2000
C = 1

print(f"C K: {C * K:.3f}")

configs = {'Underdamped': {'gamma_theta': 0.05,
                           'eta_theta': C * K,
                           'gamma_x': 0.05,
                           'eta_x': C * K},
           '"Critical"': {'gamma_theta': 0.1,
                 'eta_theta': C * K,
                 'gamma_x': 0.1,
                 'eta_x': C * K},
           'Overdamped': {'gamma_theta': 2.,
                 'eta_theta': C * K,
                 'gamma_x': 2.,
                 'eta_x': C * K},
           }

configs_plot = {'Underdamped': {'c': 'red', 'marker': '*'},
                '"Critical"': {'c': 'blue', 'marker': 's'},
                'Overdamped': {'c': 'purple', 'marker': 'p'},
                }

trainers = {'Ours': init_atrainer}

trainers_plot = {'Ours': {'linestyle': '-',
                          'markevery': 150},
                 'EM': {'linestyle': '-.'}}

# Run experiment
idx = torch.arange(n_samples)
for i in range(n_repeats):
    dataset = generator.sample(n_samples).to(device)
    # Baseline
    model = ToyHMM(dim=1).to(device)
    trainer = init_trainer(model)
    result = empty_record()

    for step in range(n_steps):
        loss = trainer.step(dataset, idx)
        result['theta'].append(trainer.model.theta.item())
        result['loss'].append(loss)
    baseline_results['parem'].append(result)

    # Proposal
    for config_name, config in configs.items():
        print(config_name, config)
        for trainer_name, trainer_init in trainers.items():
            model = ToyHMM(dim=1).to(device)
            trainer = trainer_init(model, **config)
            result = empty_record()
            for step in range(n_steps):
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
plt.savefig(save_dir / f'all.pdf')
plt.clf()
plt.close()

for qoi in empty_record().keys():
    # Plot Baseline results
    for baseline_name, baseline_results in baseline_summary.items():
        plt.plot(baseline_results[qoi],
                 label=r"\texttt{PGD}",
                 linestyle='--',
                 c='black')
    
    # Plot Truth
    if qoi == 'theta':
        plt.axhline(y=TRUTH, color='black', linestyle=':')
        plt.ylabel(r"$\theta$")
    plt.xlabel(r"$k$")
    plt.tight_layout()

    # Plot proposals
    for config, config_results in summary.items():
        for trainer_name, trainer_results in config_results.items():
            plt.plot(trainer_results[qoi],
                     label=config,
                     **trainers_plot[trainer_name],
                     **configs_plot[config],
                     alpha=0.5)
        plt.legend(loc='lower right', fontsize='16')
        ylim = 1.5 * np.max(config_results['Ours'][qoi])
        plt.ylim(0, ylim)
        plt.savefig(save_dir / f'{qoi}.pdf')
    plt.clf()
    plt.close()
