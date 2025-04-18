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
n_samples = 1
dim = 100
n_repeats = 1
n_steps = 2000

# ParEM parameters
n_particles = 100
STEP_SIZE = 1e-2
theta_step_size = STEP_SIZE / n_samples / dim
q_step_size = STEP_SIZE
t_m = 0.9
q_m = 0.9
device = 'cuda'
TRUTH = 100.

# Results record
def empty_record():
    return {'theta': [],
            'loss': []}

results = defaultdict(lambda : defaultdict(list))
baseline_results = defaultdict(list)

generator = ToyHMMProblem(theta=TRUTH, dim=dim)

def choose_eta_theta(gamma_theta,
                     momentum=0.9,
                     theta_step_size=theta_step_size,):
    gamma_theta = (1 - momentum) / gamma_theta / theta_step_size
    return gamma_theta

gamma = 0.7
# Initialize accerated trainer
def init_atrainer(model, 
                  eta_theta,
                  eta_x):
    trainer = AccParticleML_Proposed(model=model,
                                     theta_step_size=theta_step_size,
                                     q_step_size=q_step_size,
                                     train_size=n_samples,
                                     eval_size=0,
                                     n_particles=n_particles,
                                     gamma_theta=gamma,
                                     eta_theta=eta_theta,
                                     gamma_x=gamma,
                                     eta_x=eta_x,
                                     device=device)
    return trainer

def init_trainer(model):
    trainer = ParticleML(model=model,
                         theta_step_size=theta_step_size,
                         q_step_size=q_step_size,
                         train_size=n_samples,
                         eval_size=0,
                         opt_str="sgd",
                         n_particles=n_particles,
                         device=device)
    return trainer

b = 2 + dim
K = b + (dim ** 2 - 4) ** 0.5
C = 2

configs = {'x': {
                 'eta_theta':  C * K,
                 'eta_x': C * K},
           'y': {
                 'eta_theta': 0.1 * C * K,
                 'eta_x': 0.1 * C * K},
           'k': {
                 'eta_theta': 10 * C * K,
                 'eta_x': 10 * C * K},
           }

configs_plot = {'x': {'c': 'red', 'marker': '*'},
                'y': {'c': 'blue', 'marker': 's'},
                'k': {'c': 'purple', 'marker': 'p'},
                'd': {'c': 'green', 'marker': 'h' },
                }

trainers = {'Ours': init_atrainer}

trainers_plot = {'Ours': {'linestyle': '-',
                          'markevery': 200},
                 'EM': {'linestyle': '-.'}}

# Run experiment
idx = torch.arange(n_samples)
for i in range(n_repeats):
    dataset = generator.sample(n_samples).to(device)
    # Baseline
    model = ToyHMM(dim=dim).to(device)
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
            model = ToyHMM(dim=dim).to(device)
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
plt.savefig(f'./results/toyhmm2_all.pdf')
plt.clf()
plt.close()

for qoi in empty_record().keys():
    # Plot Baseline results
    for baseline_name, baseline_results in baseline_summary.items():
        plt.plot(baseline_results[qoi],
                 label=baseline_name,
                 linestyle='-',
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
                     **configs_plot[config],
                     alpha=0.5)
        plt.legend()
        ylim = 1.5 * np.max(config_results['Ours'][qoi])
        print(ylim)
        plt.ylim(0, ylim)
        plt.savefig(f'./results/toyhmm2_{qoi}.pdf')
    plt.clf()
    plt.close()
