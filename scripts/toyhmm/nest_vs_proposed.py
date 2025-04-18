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

NAME = "Nesterov_vs_Proposed"
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
STEP_SIZE = (1e-3 / dim/ n_samples) ** 0.5 
n_particles = 10
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

generator = ToyHMMProblem(theta=TRUTH)

def choose_eta_theta(gamma_theta,
                     theta_step_size,
                     momentum=0.9):
    gamma_theta = (1 - momentum) / gamma_theta / theta_step_size
    return gamma_theta

# Initialize accerated trainer
def init_nesterov_trainer(model, 
                          gamma_theta,
                          eta_theta,
                          gamma_x,
                          eta_x,
                          step_size):
    trainer = MPD_Nesterov(model=model,
                           theta_step_size=step_size,
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

def init_ourtrainer(model, 
                    gamma_theta,
                     eta_theta,
                     gamma_x,
                     eta_x,
                     step_size):
    trainer = MPD_Proposed(model=model,
                           theta_step_size=step_size,
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

def init_trainer(model, step_size):
    trainer = ParticleML(model=model,
                         theta_step_size=step_size,
                         q_step_size=q_step_size,
                         train_size=n_samples,
                         eval_size=0,
                         n_particles=n_particles,
                         device=device)
    return trainer

b = 2 + dim
K = b + (dim ** 2 - 4) ** 0.5

g_e_one =  1 / STEP_SIZE
g_e_two = 1 / STEP_SIZE
g_e_three = 1 / STEP_SIZE
m_one = 0.99
m_two = 0.9
m_three = 0.5
configs = {'x': {'gamma_theta': choose_eta_theta(g_e_one, STEP_SIZE, momentum=m_one),
                 'eta_theta': g_e_one,
                 'gamma_x': 1.,
                 'eta_x': 2 * K,
                 'step_size': STEP_SIZE},
           'y': {'gamma_theta': choose_eta_theta(g_e_two, STEP_SIZE, momentum=m_two),
                 'eta_theta': g_e_two,
                 'gamma_x': 1.,
                 'eta_x': 2 * K,
                 'step_size': STEP_SIZE},
           'z': {'gamma_theta':choose_eta_theta(g_e_three, STEP_SIZE, momentum=m_three),
                 'eta_theta':  g_e_three,
                 'gamma_x': 1.,
                 'eta_x': 2 * K,
                 'step_size': STEP_SIZE}}

configs_plot = {'x': {'c': 'red', 'marker':'o', 'markevery':150},
                'y': {'c': 'blue', 'marker':'*', 'markevery':150},
                'z': {'c': 'green', 'marker':'v', 'markevery':150}}

trainers = {
            'nesterov': init_nesterov_trainer,
            'ours': init_ourtrainer,
            }

trainers_plot = {'nesterov': {'linestyle': '--',
                              'alpha' : 0.8},
                 'ours': {'linestyle': '-',
                          'alpha' : .8}}

# Run experiment
idx = torch.arange(n_samples)
for i in range(n_repeats):
    dataset = generator.sample(n_samples).to(device)
    # Baseline
    model = ToyHMM(dim=dim).to(device)
    trainer = init_trainer(model, STEP_SIZE)
    result = empty_record()
    for step in tqdm(range(n_steps)):
        loss = trainer.step(dataset, idx)
        result['theta'].append(trainer.model.theta.item())
        result['loss'].append(loss)
    baseline_results['parem'].append(result)

    # Proposal
    for config_name, config in configs.items():
        for trainer_name, trainer_init in trainers.items():
            print(trainer_name, config)
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
plt.savefig(save_path / f'_all.pdf')
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
                     **configs_plot[config])
        plt.legend()
        plt.savefig(save_path / f'{qoi}.pdf')
    plt.clf()
    plt.close()