r"""Initial process for fixing all possible random seed.
"""

import random

import numpy as np
import torch

from ATTA.utils.config_reader import Conf


def reset_random_seed(config: Conf):
    r"""
    Initial process for fixing all possible random seed.

    Args:
       config (Conf): munchified dictionary of args (:obj:`config.random_seed`)


    """
    # Fix Random seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Default state is a training state
    torch.enable_grad()
