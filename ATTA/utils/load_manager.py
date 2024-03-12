from typing import Union, Dict

import torch.nn
from torch.utils.data import DataLoader

from ATTA import register
from ATTA.kernel.launchers.basic_launcher import Launcher
from ATTA.utils.config_reader import Conf
from ATTA.utils.initial import reset_random_seed


def load_launcher(name: str) -> Launcher:
    r"""
    A launcher loader.
    Args:
        name (str): Name of the chosen launcher

    Returns:
        A instantiated launcher.

    """
    try:
        launcher = register.launchers[name]()
    except KeyError as e:
        print(f'#E#Launcher {name} dose not exist.')
        raise e
    return launcher

def load_atta_algorithm(config: Conf):
    r"""
    A pipeline loader.
    Args:
        name (str): Name of the chosen pipeline
        config (Conf): Please refer to specific GNNs for required configs and formats.

    Returns:
        A instantiated pipeline.

    """
    try:
        reset_random_seed(config)
        pipeline = register.algs[config.atta.name](config)
    except KeyError as e:
        print(f'#E#TTA algorithm {config.atta.name} dose not exist.')
        raise e
    return pipeline
