import itertools
import os
import os.path
import sys
from pathlib import Path

from ATTA.definitions import ROOT_DIR
from ATTA.utils.load_manager import load_launcher
from ATTA.utils.args import AutoArgs
from ATTA.utils.config_reader import load_config


def launch():
    conda_interpreter = sys.executable
    conda_goodtg = os.path.join(sys.exec_prefix, 'bin', 'attatg')
    auto_args = AutoArgs().parse_args(known_only=True)
    auto_args.config_root = get_config_root(auto_args)

    jobs_group = make_list_cmds(auto_args, conda_goodtg)
    launcher = load_launcher(auto_args.launcher)
    launcher(jobs_group, auto_args)


def get_config_root(auto_args):
    if auto_args.config_root:
        if os.path.isabs(auto_args.config_root):
            config_root = Path(auto_args.config_root)
        else:
            config_root = Path(ROOT_DIR, 'configs', auto_args.config_root)
    else:
        config_root = Path(ROOT_DIR, 'configs', 'TTA_configs')
    return config_root


def make_list_cmds(auto_args, conda_goodtg):
    args_group = [
        f'{conda_goodtg} --task train --config_path {auto_args.config_root / dataset / "SimATTA.yaml"} --atta.SimATTA.cold_start 100 ' \
        f'--atta.SimATTA.el {el} --atta.SimATTA.nc_increase {k} --atta.gpu_clustering --exp_round 1 --atta.SimATTA.LE {le} ' \
        f'--atta.SimATTA.target_cluster {ic} --log_file SimATTA_{dataset}_LE{le}_IC{ic}_k{k}_el{el} --num_workers 4 '
        for dataset, el in [('VLCS', 1e-3)] # ('PACS', 1e-4),
        for k in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
        for le in [0, 1]
        for ic in [0, 1] if not (ic == 0 and k % 1 != 0)]

    return args_group


if __name__ == '__main__':
    launch()
