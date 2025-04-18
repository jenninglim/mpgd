from pathlib import Path
from importlib import import_module

def load_config(config_name: str):
    dataset, config = config_name.split('.')
    config_path = Path(f'./scripts/configs/')
    assert config_path.exists(), f"{config_name} Invalid config name."
    module = import_module(f"configs.{dataset}.{config}", str(config_path.resolve()))
    experiment = import_module(f"configs.{dataset}.setup", str(config_path.resolve()))

    print("Loading config...")
    algo_config = getattr(module, 'algo_config')
    prior_config = getattr(experiment, 'prior_config')
    model_config = getattr(experiment, 'model_config')
    dataset_config = getattr(experiment, 'dataset_config')
    experiment_config = getattr(experiment, 'experiment_config')

    return {'model_config': model_config,
            'algo_config': algo_config,
            'dataset_config': dataset_config,
            'prior_config': prior_config,
            'experiment_config': experiment_config}


def valid_config(config_name: str):
    pass