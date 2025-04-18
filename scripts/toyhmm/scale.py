from collections import defaultdict
import numpy as np
import torch
from src.models import ToyHMM
from src.problems import ToyHMMProblem
from src.algorithms.algorithms import *
from src.algorithms.other_discretizations import *
from src.preconditioners import *
from src.algorithms.vi import *
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

NAME = 'scale'
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
n_steps = 5000


X_INIT = - 5 # {-5, -20, -100} CHANGE THIS VARIABLE ACCORDINGLY
# ParEM parameters
n_particles = 10
STEP_SIZE = 1e-3
theta_step_size = STEP_SIZE 
q_step_size = STEP_SIZE
device = 'cuda'
TRUTH = 10.
TRUTH_SCALE = 12.

SAVE_PATH = Path('./results/scale')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


# Results record
def empty_record():
    return {'theta': [],
            'loss': []}

results = defaultdict(lambda : defaultdict(list))
baseline_results = defaultdict(list)

generator = ToyHMMProblem(theta=TRUTH,
                          scale=TRUTH_SCALE)

def choose_eta(gamma,
               momentum=0.9,
               step_size=STEP_SIZE,):
    gamma = (1 - momentum) / gamma / step_size
    return gamma

gamma = 0.09
# Initialize accerated trainer
def init_atrainer(model, 
                  acc_x,
                  acc_theta,):
    gamma_theta = gamma 
    gamma_x = gamma
    eta_theta = choose_eta(gamma_theta, step_size=theta_step_size)
    eta_x = choose_eta(gamma_x, step_size=q_step_size)
    precon = ConstantPreconditioner(model)
    trainer = ParticleML_One(model=model,
                             theta_step_size=theta_step_size,
                             q_step_size=q_step_size,
                             train_size=n_samples,
                             eval_size=0,
                             n_particles=n_particles,
                             gamma_theta=gamma,
                             eta_theta=eta_theta,
                             gamma_x=gamma,
                             eta_x=eta_x,
                             acc_x=acc_x,
                             acc_theta=acc_theta,
                             device=device,
                             preconditioner=precon,)
    trainer.particles = trainer.particles + X_INIT
    return trainer

def init_trainer(model):
    precon = ConstantPreconditioner(model, c=1)
    trainer = ParticleML(model=model,
                         theta_step_size=theta_step_size,
                         q_step_size=q_step_size,
                         train_size=n_samples,
                         eval_size=0,
                         n_particles=n_particles,
                         device=device,
                         preconditioner=precon,)
    trainer.particles = trainer.particles + X_INIT
    return trainer

b = 2 + n_samples
K = b + (n_samples ** 2 - 4) ** 0.5
C = 2

configs = {
            #'PGD': {'acc_x': False,
            #        'acc_theta': False},
            'TA,XG': {'acc_x': False,
                      'acc_theta': True},
            'TG,XA': {'acc_x': True,
                      'acc_theta': False},
            'TA,XA': {'acc_x': True,
                      'acc_theta': True},
           }

configs_plot = {
    #'PGD': {'c': 'red', 'marker': '*'},
    'TA,XG': {
        'c': 'blue',
        #'marker': 's'
        },
    'TG,XA': {
        'c': 'purple',
        #'marker': 'p'
        },
    'TA,XA': {
        'c': 'green',
        #'marker': 'h'
        },
}

trainers = {'Ours': init_atrainer}

trainers_plot = {'Ours': {'linestyle': '-',
                          'markevery': 200},
                 'EM': {'linestyle': '-.'}}

# Run experiment
idx = torch.arange(n_samples)
for i in range(n_repeats):
    dataset = generator.sample(n_samples).to(device)
    dataset = dataset - dataset.mean() + TRUTH 
    # Baseline
    model = ToyHMM(dim=1,
                   scale=TRUTH_SCALE).to(device)
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
            model = ToyHMM(dim=1,
                           scale=TRUTH_SCALE).to(device)
            trainer = trainer_init(model, **config)
            result = empty_record()
            for step in range(n_steps):
                loss = trainer.step(dataset, idx)
                result['theta'].append(trainer.model.theta.item())
                result['loss'].append(loss)
            results[config_name][trainer_name].append(result)
    
# Init summary
mu_summary = defaultdict(dict)
std_summary = defaultdict(dict)
bl_mu_summary = {}
bl_std_summary = {}

# Calculate the mean of baseline
for baseline_name, baseline_result in baseline_results.items():
    losss = [result['loss'] for result in baseline_result]
    thetas = [result['theta'] for result in baseline_result]

    theta_mu = np.mean(np.stack(thetas, axis=0), axis=0)
    loss_mu = np.mean(np.stack(losss, axis=0), axis=0)
    theta_std = np.std(np.stack(thetas, axis=0), axis=0)
    loss_std = np.std(np.stack(losss, axis=0), axis=0)

    bl_mu_summary[baseline_name] = {'theta': theta_mu, 'loss': loss_mu}
    bl_std_summary[baseline_name] = {'theta': theta_std, 'loss': loss_std}


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
        theta_std = np.std(thetas, axis=1)
        loss_mu = np.mean(losss, axis=1)
        loss_std = np.std(losss, axis=1)
        mu_summary[config][trainer_name] = {'theta': theta_mu, 'loss': loss_mu}
        std_summary[config][trainer_name] = {'theta': theta_std, 'loss': loss_std}
plt.savefig(SAVE_PATH / 'all.pdf')
plt.clf()
plt.close()

for qoi in empty_record().keys():
    # Plot Baseline results
    for baseline_name, baseline_results in bl_mu_summary.items():
        y = baseline_results[qoi]
        x = range(len(y))
        e = bl_std_summary[baseline_name][qoi]
        plt.errorbar(x, y, e,
                 label=baseline_name,
                 linestyle='-',
                 c='black')
    
    # Plot Truth
    if qoi == 'theta':
        plt.axhline(y=TRUTH, color='black', linestyle=':')
    if qoi == 'scale':
        plt.axhline(y=TRUTH_SCALE, color='black', linestyle=':')

    # Plot proposals
    for config, config_results in mu_summary.items():
        for trainer_name, trainer_results in config_results.items():
            y = trainer_results[qoi]
            x = range(len(y))
            e = std_summary[config][trainer_name][qoi]
            plt.errorbar(x, y, e,
                         label=trainer_name+'_'+config,
                         **trainers_plot[trainer_name],
                         **configs_plot[config],
                         alpha=1.)
        # plt.legend()
        plt.ylim(-5, 11)
        plt.xlim(0, len(y))
        plt.ylabel(r"$\theta$")
        plt.xlabel(r"$k$")
        plt.tight_layout()
        plt.legend()
        plt.savefig(SAVE_PATH / f'scale_{qoi}_{X_INIT}.pdf')
    plt.clf()
    plt.close()
