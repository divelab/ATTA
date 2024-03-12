r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import time
from typing import Tuple, Union

import torch.nn
from torch.utils.data import DataLoader

from ATTA import config_summoner
# from ATTA.utils.config_reader import config_summoner
from ATTA.utils.load_manager import load_atta_algorithm
from ATTA.utils.args import args_parser
from ATTA.utils.config_reader import Conf
from ATTA.utils.initial import reset_random_seed
from ATTA.utils.logger import load_logger
from ATTA.definitions import OOM_CODE
import multiprocessing as mp

def main():
    #
    args = args_parser()
    config = config_summoner(args)
    if config.mp_spawn:
        # torch.set_num_threads(5)
        mp.set_start_method('spawn') # ImageFolder and Subprocess may cause deadlock with multiprocessing fork
    load_logger(config)

    alg = load_atta_algorithm(config)
    tik = time.time()
    alg()
    print(f"Time cost: {time.time() - tik}s")

if __name__ == '__main__':
    main()
