import copy
import os
import shlex
import shutil
from pathlib import Path
import distutils.dir_util

import numpy as np
from ruamel.yaml import YAML
from tqdm import tqdm

from ATTA import config_summoner
from ATTA import register
from ATTA.definitions import ROOT_DIR, STORAGE_DIR
from ATTA.utils.args import AutoArgs
from ATTA.utils.args import args_parser
from ATTA.utils.config_reader import load_config, args2config, merge_dicts
from .basic_launcher import Launcher
import matplotlib.pyplot as plt


@register.launcher_register
class SensitiveLauncher(Launcher):
    def __init__(self):
        super(SensitiveLauncher, self).__init__()
        self.watch = True
        self.pick_reference = [-1, -2]
        self.hparam = 'E'
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 12

    def __call__(self, jobs_group, auto_args: AutoArgs):
        result_dict = self.harvest_all_fruits(jobs_group)
        best_fruits = self.picky_farmer(result_dict)

        if auto_args.sweep_root:
            self.process_final_root(auto_args)
            self.update_best_config(auto_args, best_fruits)

    def update_best_config(self, auto_args, best_fruits):
        for ddsa_key in best_fruits.keys():
            final_path = (auto_args.final_root / '/'.join(ddsa_key.split(' '))).with_suffix('.yaml')
            top_config, _, _ = load_config(final_path, skip_include=True)
            whole_config, _, _ = load_config(final_path)
            args_list = shlex.split(best_fruits[ddsa_key][0])
            args = args_parser(['--config_path', str(final_path)] + args_list)
            args2config(whole_config, args)
            args_keys = [item[2:] for item in args_list if item.startswith('--')]
            # print(args_keys)
            modified_config = self.filter_config(whole_config, args_keys)
            # print(top_config)
            # print(modified_config)
            final_top_config, _ = merge_dicts(top_config, modified_config)
            # print(final_top_config)
            yaml = YAML()
            yaml.indent(offset=2)
            # yaml.dump(final_top_config, sys.stdout)
            yaml.dump(final_top_config, final_path)

    def process_final_root(self, auto_args):
        if auto_args.final_root is None:
            auto_args.final_root = auto_args.config_root
        else:
            if os.path.isabs(auto_args.final_root):
                auto_args.final_root = Path(auto_args.final_root)
            else:
                auto_args.final_root = Path(ROOT_DIR, 'configs', auto_args.final_root)
        if auto_args.final_root.exists():
            ans = input(f'Overwrite {auto_args.final_root} by {auto_args.config_root}? [y/n]')
            while ans != 'y' and ans != 'n':
                ans = input(f'Invalid input: {ans}. Please answer y or n.')
            if ans == 'y':
                distutils.dir_util.copy_tree(str(auto_args.config_root), str(auto_args.final_root))
            elif ans == 'n':
                pass
            else:
                raise ValueError(f'Unexpected value {ans}.')
        else:
            shutil.copytree(auto_args.config_root, auto_args.final_root)

    def picky_farmer(self, result_dict):
        best_fruits = dict()
        sorted_fruits = dict()
        for ddsa_key in result_dict.keys():
            for key, value in result_dict[ddsa_key].items():
                result_dict[ddsa_key][key] = np.stack([np.mean(value, axis=1), np.std(value, axis=1)], axis=1)[-3]
            # lambda x: x[1][?, 0]  - ? denotes the result used to choose the best setting.
            if self.watch:
                sorted_fruits[ddsa_key] = sorted(list(result_dict[ddsa_key].items()), key=lambda x: x[0])
            else:
                best_fruits[ddsa_key] = max(list(result_dict[ddsa_key].items()), key=lambda x: sum(x[1][i, 0] for i in self.pick_reference))
            # best_fruits[ddsa_key] = sorted_fruits[ddsa_key][0]
        if self.watch:
            print(sorted_fruits)
            for ddsa_key in result_dict.keys():
                dataset_key = ddsa_key.split(' ')[0]
                ERM_results = {'GOODMotif': 60.93, 'GOODTwitter': 57.04, 'GOODHIV': 70.37}
                x = []
                y1 = []
                y2 = []
                for item in sorted_fruits[ddsa_key]:
                    x.append(item[0])
                    y1.append(item[1][0] - item[1][1])
                    y2.append(item[1][0] + item[1][1])
                y1 = np.array(y1) * 100
                y2 = np.array(y2) * 100
                standard_ticks = np.arange(len(x))
                fig, ax = plt.subplots(dpi=300)
                ax.fill_between(standard_ticks, y1, y2, alpha=.5, linewidth=0)
                ax.axhline(y=ERM_results[dataset_key], color='r', linestyle='--', label='ERM')
                ax.plot(standard_ticks, (y1 + y2) / 2, '.-', linewidth=2, label='LECI')
                ax.set(xticks=standard_ticks, xticklabels=x)
                ax.set_ylabel('Test metric', fontsize=22)
                ax.set_xlabel(f'$\lambda_\mathtt{{{self.hparam}}}$', fontsize=22)
                ax.legend(loc=1, prop={'size': 16})
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0, hspace=0)
                # plt.show()
                save_path = os.path.join(STORAGE_DIR, 'figures', 'sensitive_study', f'{dataset_key}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig.savefig(os.path.join(save_path, f'{self.hparam}.png'))
            exit(0)
        print(best_fruits)
        return best_fruits

    def filter_config(self, config: dict, target_keys):
        new_config = copy.deepcopy(config)
        for key in config.keys():
            if type(config[key]) is dict:
                new_config[key] = self.filter_config(config[key], target_keys)
                if not new_config[key]:
                    new_config.pop(key)
            else:
                if key not in target_keys:
                    new_config.pop(key)
        return new_config

    def harvest_all_fruits(self, jobs_group):
        all_finished = True
        result_dict = dict()
        for cmd_args in tqdm(jobs_group, desc='Harvesting ^_^'):
            args = args_parser(shlex.split(cmd_args)[1:])
            config = config_summoner(args)
            last_line = self.harvest(config.log_path)

            if not last_line.startswith('INFO: ChartInfo'):
                print(cmd_args, 'Unfinished')
                all_finished = False
                continue
            result = last_line.split(' ')[2:]
            num_result = len(result)
            key_args = shlex.split(cmd_args)[1:]
            round_index = key_args.index('--exp_round')
            key_args = key_args[:round_index] + key_args[round_index + 2:]

            config_path_index = key_args.index('--config_path')
            key_args.pop(config_path_index)  # Remove --config_path
            config_path = Path(key_args.pop(config_path_index))  # Remove and save its value
            config_path_parents = config_path.parents
            dataset, domain, shift, algorithm = config_path_parents[2].stem, config_path_parents[1].stem, \
                                                config_path_parents[0].stem, config_path.stem
            ddsa_key = ' '.join([dataset, domain, shift, algorithm])
            if ddsa_key not in result_dict.keys():
                result_dict[ddsa_key] = dict()

            L_E = 1 if self.hparam == 'L' else 3
            key_str = float(key_args[key_args.index('--extra_param') + L_E])  # +1: LA +3: EA
            if key_str not in result_dict[ddsa_key].keys():
                result_dict[ddsa_key][key_str] = [[] for _ in range(num_result)]
            # print(f'{ddsa_key}_{key_str}: {result}')
            result_dict[ddsa_key][key_str] = [r + [eval(result[i])] for i, r in
                                              enumerate(result_dict[ddsa_key][key_str])]
        # if not all_finished:
        #     print('Please launch unfinished jobs using other launchers before harvesting.')
        #     exit(1)
        return result_dict

