import sys
sys.path.append('.')
from configs.utils import load_config
from math import ceil
import torch
from torch.utils.data import Subset
from src import ALGORITHMS
from src.models import VAE
from src.priors import *
from src.algorithms.algorithms import *
from src.utils import (dataset_with_indices,
                       show_images)
import wandb
import typer
from torchvision.transforms.functional import to_pil_image
from configs import DATASETS
from scripts.utils import (set_seed,
                           get_seeded_dl,
                           get_best_model_path,
                           reconstruct,
                           pytorch_fid,
                           interpolate)
import optuna
from optuna.trial import TrialState
from ignite.engine import Engine, Events
from ignite.metrics import Average, FID
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine, ProgressBar
from ignite.handlers import EarlyStopping, Checkpoint
import torchvision.transforms as transforms
import json

DIAGNOSTIC_EVERY_X = 5

app = typer.Typer()


def _make_dataset_transform(dataset_config):
    transform = [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),]
    transform.append(transforms.Lambda(lambda x: x.view(-1)))
    if dataset_config['dequantize']:
        transform.append(transforms.Lambda(lambda x: x + torch.rand_like(x) / 256))
    transform.append(transforms.Lambda(lambda x: torch.clip(x, min=-1., max=1)))
    transform = transforms.Compose(transform)
    return transform

def _make_dataset(dataset_config, additional=None, test_only=False):
    transform = _make_dataset_transform(dataset_config)
    test_dataset = DATASETS[dataset_config['dataset']](root='data',
                                                       download=True,
                                                       train=False,
                                                       transform=transform)
    test_datasetidx = dataset_with_indices(Subset)(test_dataset,
                                                range(dataset_config['n_test']))
    if additional is not None:
        additional = dataset_with_indices(Subset)(test_dataset,
                                                    range(dataset_config['n_mse']))
    if test_only:
        if additional is not None:
            return test_datasetidx, additional
        return test_datasetidx
    # Load data
    train_dataset = DATASETS[dataset_config['dataset']](root='data',
                                                        download=True,
                                                        train=True,
                                                        transform=transform)
    train_indices = range(dataset_config['n_train'])
    train_subset = dataset_with_indices(Subset)(train_dataset,
                                                train_indices)


    eval_indices = range(dataset_config['n_train'],
                         dataset_config['n_train'] + dataset_config['n_eval'])
    eval_subset = dataset_with_indices(Subset)(train_dataset, eval_indices)
    return train_subset, eval_subset, test_dataset


def _make_algo(algo_config, model, train_size, eval_size, device):
    assert algo_config['algorithm'] in ALGORITHMS.keys()
    algo = ALGORITHMS[algo_config['algorithm']](model=model,
                                                **algo_config,
                                                train_size=train_size,
                                                eval_size=eval_size,
                                                device=device)
    return algo


def _make_model(model_config,
                prior_config,
                y_dim,
                n_channels,
                device):
    # Instantiate Model
    # VAMPVarPrior vs VAMPCovPrior
    if prior_config['prior_type'] == 'vamp':
        if prior_config['cov_type'] == 'diag':
            prior = VAMPVarPrior(**prior_config,
                                device=device)
        elif prior_config['cov_type'] == 'full':
            prior = VAMPVarPrior(**prior_config,
                                device=device)
        else:
            raise ValueError(f"Invalid cov_type: {prior_config['cov_type']}")
    elif prior_config['prior_type'] == 'flow':
        prior = FlowPrior(**prior_config,
                          device=device)

    model = VAE(**model_config,
                prior=prior,
                x_dim=prior_config['x_dim'],
                y_dim=y_dim,
                n_channels=n_channels,
                device=device) 
    return model


def _diagnostics_model_samples(model, dataset_config, n_samples=25):
    y, x = model.sample(n_samples)
    sqrt_y_dim = int(dataset_config['y_dim'] ** 0.5)
    y = y.reshape(n_samples,
                    dataset_config['n_channels'],
                    sqrt_y_dim,
                    sqrt_y_dim)
    # rescale
    y = (y + 1) / 2
    y = y.clip(0, 1)
    fig, grid = show_images(y, show=False, nrow=int(n_samples ** 0.5))
    return grid 


def _diagnostics(model, algo, dataset_config, train_subset, eval_subset):
    transform = _make_dataset_transform(dataset_config)
    results = {}
    sqrt_y_dim = int(dataset_config['y_dim'] ** 0.5)
    grid = _diagnostics_model_samples(model,
                                      dataset_config)
    wimg = wandb.Image(grid)
    results['sample'] = wimg

    if hasattr(algo, 'particles'):
        train_indices = [3, 5, 10]
        eval_indices = [3, 5, 10]
        eval_indices = [ind + len(train_subset) for ind in eval_indices]
        view_n_particles = 3
        train_true_ys = [transform(to_pil_image(img)) for img in train_subset.dataset.data[train_indices]]
        eval_true_ys = [transform(to_pil_image(img)) for img in eval_subset.dataset.data[eval_indices]]
        true_ys = torch.stack(train_true_ys + eval_true_ys, dim=0)
        true_ys = true_ys.reshape(
            len(train_indices) + len(eval_indices),
            1,
            dataset_config['n_channels'],
            sqrt_y_dim,
            sqrt_y_dim
        )

        particle_indices = train_indices + eval_indices
        particle_ys = model.decoder(algo.particles[particle_indices, :view_n_particles]).reshape(
            len(train_indices) + len(eval_indices),
            -1,
            dataset_config['n_channels'],
            sqrt_y_dim,
            sqrt_y_dim
        )
        stacked = torch.concat([true_ys.cpu(),
                                particle_ys.cpu()], dim=1)
        stacked = stacked.flatten(0,1)
        stacked = (stacked + 1) / 2
        fig, grid = show_images(stacked,
                                show=False,
                                nrow=view_n_particles + 1)
        rimg = wandb.Image(grid)
        results['reconstruction'] = rimg
    return results


def _get_generator_step(model, dataset_config):
    sqrt_y_dim = int(dataset_config['y_dim'] ** 0.5)
    def eval_fid_step(engine, batch):
        with torch.no_grad():
            (real_imgs, _), idx = batch
            n_batch = real_imgs.shape[0]
            gen_imgs, _ = model.sample(n_batch)
            gen_imgs = gen_imgs.reshape(n_batch,
                                        dataset_config['n_channels'],
                                        sqrt_y_dim,
                                        sqrt_y_dim)
            real_imgs = real_imgs.reshape(n_batch,
                                        dataset_config['n_channels'],
                                        sqrt_y_dim,
                                        sqrt_y_dim)
            gen_imgs = interpolate(gen_imgs)
            real_imgs = interpolate(real_imgs)
        return real_imgs, gen_imgs
    return eval_fid_step


def _train(algo_config,
           dataset_config,
           model_config,
           prior_config,
           experiment_config,
           config_name,
           wandb_log=False,
           optuna_trial=None):
    seed = experiment_config['seed']
    device = experiment_config['device']

    set_seed(seed)

    DataLoader = get_seeded_dl(seed)

    # Instantiate Model
    model = _make_model(model_config,
                        prior_config=prior_config,
                        y_dim=dataset_config['y_dim'],
                        n_channels=dataset_config['n_channels'],
                        device=device)

    if experiment_config['preload']:
        print(f"Preloading")
        checkpoint = torch.load(f"checkpoint/{dataset_config['dataset']}_preload.pt")
        decoder_state_dict = {}
        for key, params in checkpoint.items():
            if key.split(".")[0] == "decoder":
                decoder_state_dict[key[8:]] = params
        model.decoder.load_state_dict(decoder_state_dict)
    model.eval()

    # Instantiate Dataset
    train_subset, eval_subset, _ = _make_dataset(dataset_config)

    # Instantiate Dataloader
    train_dl = DataLoader(train_subset,
                          batch_size=experiment_config['batch_size'])
    eval_dl = DataLoader(eval_subset,
                         batch_size=experiment_config['batch_size'])
    particle_subset = Subset(eval_subset, range(experiment_config['particle_subset'])) 
    particle_dl = DataLoader(particle_subset,
                             batch_size=experiment_config['batch_size'])
    # Instantiate Algorithm
    algo = _make_algo(algo_config,
                      model,
                      len(train_subset),
                      len(eval_subset),
                      device)

    def train_step(engine, batch):
        algo.model.train()
        (imgs, _), idx = batch
        loss = algo.step(imgs.to(device), idx)
        algo.model.eval()
        assert not np.isnan(loss), f"NaN detected in loss {loss}"
        return {'loss' : loss}

    def train_statistics(engine, batch):
        (imgs, _), idx = batch
        algo.model.eval()
        loss = algo._eval_loss(imgs.to(device), idx)
        return {'train_loss' : loss}

    def validation_step(engine, batch):
        (imgs, _), idx = batch
        eval_idx = len(train_subset) + idx
        if hasattr(algo, 'particles'):
            algo._catchup_particles(imgs.to(device), eval_idx)
            before_eval_loss = algo._eval_loss(imgs.to(device), eval_idx)
            algo._step_q(imgs.to(device), eval_idx)
            eval_loss = algo._eval_loss(imgs.to(device), eval_idx)
            prior_log_p = algo.model.prior.log_prob(algo.particles[eval_idx]).mean()
        else:
            eval_loss = 0
            before_eval_loss = 0
            prior_log_p = 0
        if np.isnan(eval_loss):
            eval_loss = algo._eval_loss(imgs.to(device), eval_idx)
            print(f"particles nan # {algo.particles[eval_idx].isnan().sum()}",)
            if hasattr(algo, 'qmo'):
                print(f"momentuim nan # {algo.qmo[eval_idx].isnan().sum()}")  
            print("In Eval NaN detected")
            algo.reset_particles(eval_idx)
        return {'eval_loss': eval_loss,
                'before_eval_loss': before_eval_loss,
                'prior_log_p': prior_log_p}

    trainer = Engine(train_step)
    eval_eval = Engine(validation_step)
    gen_step = _get_generator_step(model, dataset_config)
    train_eval = Engine(train_statistics)

    fid = pytorch_fid(device=device)
    gen_eval= Engine(gen_step)
    fid.attach(gen_eval, 'fid')

    pbar = ProgressBar()
    pbar.attach(gen_eval)
    def score_function(engine):
        return - engine.state.metrics["fid"]

    model_checkpoint = ModelCheckpoint(
        f"checkpoint/{config_name}",
        n_saved=2,
        filename_prefix=f"best_seed_{seed}",
        score_function=score_function,
        require_empty=False,
        score_name="loss",
        global_step_transform=global_step_from_engine(trainer), # helps fetch the trainer's state
    )

    early_stopping_handler = EarlyStopping(patience=experiment_config['early_stop'],
                                           score_function=score_function,
                                           trainer=trainer)
    gen_eval.add_event_handler(Events.COMPLETED,
                                     early_stopping_handler)
    gen_eval.add_event_handler(Events.COMPLETED,
                                    model_checkpoint,
                                    {"model": model})

    # Attach metrics
    eval_loss_avg = Average(output_transform=lambda output: output['eval_loss'])
    train_loss_avg = Average(output_transform=lambda output: output['train_loss'])
    eval_plogp_avg = Average(output_transform=lambda output: output['prior_log_p'])
    beval_plogp_avg = Average(output_transform=lambda output: output['before_eval_loss'])

    train_loss_avg.attach(train_eval, 'loss')
    eval_loss_avg.attach(eval_eval, 'loss')
    eval_plogp_avg.attach(eval_eval, 'prior_log_p')
    beval_plogp_avg.attach(eval_eval, 'before_eval_loss')

    if experiment_config['pretrain'] > 0:
        model.prior.set_pretrain()

    # End pretraining
    if experiment_config['pretrain'] > 0:
        @trainer.on(Events.ITERATION_STARTED(once=experiment_config['pretrain']))
        def switch_pretrain(trainer):
            print("Ending Pretraining")
            model.prior.set_nopretrain()

    # Compute end of epoch statistics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        model.eval()
        if 'loss' in train_eval.state.metrics.keys():
            prev_loss = train_eval.state.metrics['loss']
        else:
            prev_loss = float('inf')
        with torch.no_grad():
            eval_eval.run(particle_dl)
            gen_eval.run(eval_dl)
            train_eval.run(train_dl)
        train_metrics = train_eval.state.metrics
        eval_metrics = eval_eval.state.metrics
        gen_metrics = gen_eval.state.metrics
        print(f"""Epoch: {trainer.state.epoch};
               i: {trainer.state.iteration};
               Train Logp: {train_metrics['loss']:.2f};
               Eval Logp: {eval_metrics['loss']:.2f};
               Before Eval Logp: {eval_metrics['before_eval_loss']:.2f};
               FID: {gen_metrics['fid']:.2f};""")
        
        if np.isnan(train_metrics['loss']) or np.isnan(eval_metrics['loss']):
            print(eval_metrics)
            #raise Exception("NaN detected, stopping training")

        if wandb_log:
            wandb.log(
                {**{f'train_{k}': v for k, v in  train_metrics.items()},
                 **{f'eval_{k}': v for k, v in eval_metrics.items()},
                 **gen_metrics},
                step=trainer.state.epoch)
        if optuna_trial is not None:
            optuna_trial.report(eval_metrics['loss'], trainer.state.epoch+1)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    # Enabling wandb logging.
    if wandb_log:
        @trainer.on(Events.EPOCH_COMPLETED(every=DIAGNOSTIC_EVERY_X))
        def wandb_log(trainer):
            model.eval()
            with torch.no_grad():
                diags = _diagnostics(model,
                                     algo,
                                     dataset_config,
                                     train_subset,
                                     eval_subset)
                wandb.log(diags, step=trainer.state.epoch+1)
    
    # On completion
    @trainer.on(Events.COMPLETED)
    def print_ending(trainer):
        print("Training Complete")
        print(f"Best model at epoch {trainer.state.epoch} {trainer.state.iteration}")
        print(f"Best model loss {early_stopping_handler.best_score}")
    
    pbar = ProgressBar()
    pbar.attach(trainer)
    
    trainer.run(train_dl,
                max_epochs=experiment_config['n_updates'] / len(train_dl))
    eval_loss = eval_eval.state.metrics['loss']
    return eval_loss


@app.command()
def train(config_name='example_1', seed=None):
    configs = load_config(config_name)

    algo_config = configs['algo_config']
    dataset_config = configs['dataset_config']
    model_config = configs['model_config']
    prior_config = configs['prior_config']
    experiment_config = configs['experiment_config']
    if seed is not None:
        print("Overriding seed")
        experiment_config['seed'] = int(seed)
    
    wandb.init(
        project=dataset_config['dataset'],
        config={
            **prior_config,
            **dataset_config,
            **model_config,
            **algo_config,
            **experiment_config,
        }
    )
    _train(algo_config,
           dataset_config,
           model_config,
           prior_config,
           experiment_config,
           config_name,
           wandb_log=True,
           optuna_trial=None)


@app.command()
def tune(n_trials=20,
         config_name='example_1'):
    n_trials = int(n_trials)
    configs = load_config(config_name)
    db_name = config_name + f"{configs['dataset_config']['n_train']}"
    study = optuna.create_study(direction='minimize',
                                storage=f"sqlite:///db.sqlite3",
                                study_name=db_name,
                                load_if_exists=True)

    def objective(trial):
        configs = load_config(config_name)

        algo_config = configs['algo_config']
        dataset_config = configs['dataset_config']
        model_config = configs['model_config']
        prior_config = configs['prior_config']
        experiment_config = configs['experiment_config']

        ## Use Optuna's suggest API
        algo_config['gamma_theta'] = trial.suggest_float('gamma_theta',
                                                         0.1,
                                                         10.)

        algo_config['gamma_x'] = trial.suggest_float('gamma_x',
                                                     0.1,
                                                     10.)

        algo_config['eta_theta'] = trial.suggest_float('eta_theta',
                                                        1.,
                                                        1000.,)

        algo_config['eta_x'] = trial.suggest_float('eta_x',
                                                    1.,
                                                    1000.)

        train_loss = _train(algo_config,
                            dataset_config,
                            model_config,
                            prior_config,
                            experiment_config,
                            config_name,
                            wandb_log=False,
                            optuna_trial=trial)
        return train_loss
    
    study.optimize(objective, n_trials=n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


@app.command()
def eval(config_name='example_1', seed=None, samples_only:bool=False):
    Path("results").mkdir(exist_ok=True)
    configs = load_config(config_name)

    algo_config = configs['algo_config']
    dataset_config = configs['dataset_config']
    model_config = configs['model_config']
    prior_config = configs['prior_config']
    experiment_config = configs['experiment_config']

    device = experiment_config['device']
    if seed is not None:
        print("Overriding seed")
        experiment_config['seed'] = int(seed)
    seed = experiment_config['seed']
    checkpoint_file = get_best_model_path(config_name, seed)

    set_seed(seed)
    test_subset, mse_subset = _make_dataset(
        dataset_config,
        additional=dataset_config['n_mse'],
        test_only=True
        )

    # Instantiate Dataloader
    test_dl = DataLoader(test_subset,
                         batch_size=32)
    mse_dl = DataLoader(mse_subset,
                         batch_size=32)

    model = _make_model(model_config,
                        prior_config,
                        dataset_config['y_dim'],
                        dataset_config['n_channels'],
                        device)
    results = {}

    print(f"Loading {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file)
    Checkpoint.load_objects(to_load={'model': model},
                            checkpoint=checkpoint)
    
    model.eval()
    
    img = _diagnostics_model_samples(model, dataset_config, n_samples=64)
    img.save(f"results/{config_name}_{seed}_sample.pdf")
    if samples_only:
        exit()

    gen_eval_step = _get_generator_step(model, dataset_config)

    sqrt_y_dim = int(dataset_config['y_dim'] ** 0.5)

    pbar = ProgressBar()
    fid = pytorch_fid(device=device)
    gen_eval= Engine(gen_eval_step)
    fid.attach(gen_eval, 'fid')
    pbar.attach(gen_eval)
    gen_eval.run(test_dl)
    results['fid'] = gen_eval.state.metrics['fid']

    ula_n_steps = 500
    q_step_size = 1e-3
    n_particles = 5
    reconstruct_imgs = lambda imgs : reconstruct(
        imgs,
        model,
        q_step_size,
        n_particles,
        ula_n_steps
    )

    sample_imgs = torch.stack([test_subset[i][0][0] for i in range(5)], dim=0)
    sample_rimgs = reconstruct_imgs(sample_imgs)

    sample_imgs_w_recon = torch.cat([sample_imgs.unsqueeze(1),
                                     sample_rimgs], dim=1).flatten(0,1)
    sample_imgs_w_recon = (sample_imgs_w_recon + 1) / 2
    sample_imgs_w_recon = sample_imgs_w_recon.reshape(-1,
                                                      dataset_config['n_channels'],
                                                      sqrt_y_dim,
                                                      sqrt_y_dim)
    sample_imgs_pil = show_images(sample_imgs_w_recon,
                                  nrow=6)[1]
    sample_imgs_pil.save(f"results/{config_name}_recon.pdf")

    def mse_eval_step(engine, batch):
        (imgs, _), idx = batch
        with torch.no_grad():
            rimgs = reconstruct_imgs(imgs)
            mse = ((imgs.unsqueeze(1) - rimgs) ** 2).mean([-1]).mean(0).mean(0).item()
        return mse

    pbar = ProgressBar()
    mse_eval = Engine(mse_eval_step)
    pbar.attach(mse_eval)
    average = Average()
    average.attach(mse_eval, 'mse')
    mse_eval.run(mse_dl)
    results['mse'] = mse_eval.state.metrics['mse']
    print(results)

    with open(f"results/{config_name}_{seed}_eval.json", 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    app()
