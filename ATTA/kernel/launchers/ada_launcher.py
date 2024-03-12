import subprocess
import os
import shlex
import subprocess
import time

import psutil
import pynvml
from tqdm import trange

from ATTA import register
from ATTA.definitions import OOM_CODE
from .basic_launcher import Launcher
from threading import Thread


@register.launcher_register
class AdaLauncher(Launcher):
    def __init__(self):
        super(AdaLauncher, self).__init__()
        self.initial_aggressive = 80
        self.cpu_use_limit = 90
        self.ram_use_limit = 80
        self.cpu_max_wait = 120
        self.gpu_use_limit = 90
        self.num_process_limit = 1

        self.summary_string = ''
        self.allow_devices = []

    def __call__(self, jobs_group, auto_args):
        jobs_group = super(AdaLauncher, self).__call__(jobs_group, auto_args)
        pynvml.nvmlInit()
        print("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
        device_count = pynvml.nvmlDeviceGetCount()
        print("Number of devices:", device_count)
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
        handles = list(enumerate(handles))

        self.allow_devices = auto_args.allow_devices
        _thread = Thread(target=self.change_device, daemon=True).start()

        aggressive = {cmd_args: self.initial_aggressive for cmd_args in jobs_group}
        total_num_args = jobs_group.__len__()

        process_pool = {}
        jobs_status = {'done': [], 'failed': []}
        while jobs_group:
            # process_pool: in progress process (this program thinks)
            # jobs_group: processes that didn't finish successfully
            #   including in progress Process and failure processes
            #   when it is empty: all processes finished

            # --- Process emit ---
            summary_leave = False
            for check_count, cmd_args in enumerate(jobs_group):
                self.summary_string = f'Waiting: {len(jobs_group) - len(process_pool.keys())} - In progress: {len(process_pool.keys())} ' \
                                 f'- Finished: {len(jobs_status["done"])} - Failed: {len(jobs_status["failed"])}'

                if cmd_args in process_pool.keys():
                    continue

                # --- check cpu usage ---
                wait_count = self.wait_cpu()
                if wait_count >= self.cpu_max_wait:
                    print(f'\r{self.summary_string}| Wait too long, check process.', end='')
                    break

                print(f'\r{self.summary_string}| CPU/RAM available.', end='')

                for cur_i, (device_idx, device_handle) in enumerate(handles):
                    meminfo, usageinfo = self.get_gpu_info(device_handle, device_idx)
                    # fork children

                    if device_idx in self.allow_devices \
                            and meminfo < aggressive[cmd_args] \
                            and usageinfo < self.gpu_use_limit \
                            and len(process_pool.keys()) < self.num_process_limit:
                        print(f'\n\033[1;34mEmit\033[0m process on device:{device_idx}:\n{cmd_args}')

                        process = subprocess.Popen(shlex.split(cmd_args) + ['--gpu_idx', f'{device_idx}'],
                                                   close_fds=True,
                                                   stdout=open(os.devnull, 'w'),
                                                   stderr=open(os.devnull, 'w'),
                                                   cwd=os.getcwd(),
                                                   env=os.environ,
                                                   start_new_session=False)
                        process_pool[cmd_args] = process

                        for _ in trange(10, desc=f'{self.summary_string}| Interval...', leave=False):
                            time.sleep(1)
                        break
                handles.append(handles.pop(cur_i))
                if (check_count + 1) % 50 == 0 or len(process_pool.keys()) >= self.num_process_limit:
                    break

            # --- Process check ---
            ceased_processes = []
            ceased_exist = False
            for cmd_args, process in process_pool.items():

                return_code = process.poll()
                if return_code is not None:
                    if not ceased_exist:
                        print('')
                        ceased_exist = True

                    # process ceased
                    ceased_processes.append(cmd_args)

                    if return_code == 0:
                        print(f'\033[1;32mFinished\033[0m:{cmd_args}')
                        jobs_group.remove(cmd_args)
                        jobs_status['done'].append(cmd_args)
                    elif return_code == OOM_CODE:
                        if aggressive[cmd_args] > 15:
                            aggressive[cmd_args] -= 10
                            print(
                                f'\033[1;33mAbort\033[0m process:{cmd_args} due to CUDA out of memory. [decrease aggressive: {aggressive[cmd_args]}]')
                        else:
                            print(
                                f'\033[1;31mAbort\033[0m process:{cmd_args} due to CUDA memory not enough. Return code: {return_code}')
                            jobs_group.remove(cmd_args)
                            jobs_status['failed'].append(cmd_args)
                    else:
                        print(
                            f'\033[1;31mAbort\033[0m process:{cmd_args} due to other issues. Return code: {return_code}')
                        jobs_group.remove(cmd_args)
                        jobs_status['failed'].append(cmd_args)

            for ceased_process in ceased_processes:
                process_pool.pop(ceased_process)

            # --- Temporary summary ---
            self.summary_string = f'Waiting: {len(jobs_group) - len(process_pool.keys())} - In progress: {len(process_pool.keys())} ' \
                             f'- Finished: {len(jobs_status["done"])} - Failed: {len(jobs_status["failed"])}'
            for _ in trange(20, desc=f'{self.summary_string}| Waiting for emit...', leave=summary_leave):
                time.sleep(1)

    def wait_cpu(self):
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        wait_count = 0
        available_count = 0
        while wait_count < self.cpu_max_wait:
            if cpu_percent < self.cpu_use_limit and ram_percent < self.ram_use_limit:
                available_count += 1
            else:
                available_count = 0
            if available_count >= 2:
                break
            print(f'\r{self.summary_string}| Waiting for cpu/ram: {cpu_percent}/{ram_percent}', end='')
            time.sleep(1)
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            wait_count += 1
        return wait_count

    def get_gpu_info(self, device_handle, device_idx):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
        meminfo = meminfo.used / meminfo.total * 100
        usageinfo = []
        for _ in range(5):
            usageinfo.append(pynvml.nvmlDeviceGetUtilizationRates(device_handle).gpu)
            time.sleep(0.1)
        usageinfo = max(usageinfo)
        print(f'\r{self.summary_string}| Try device {device_idx} usage/mem: {usageinfo}/{meminfo:.2f}', end='')
        return meminfo, usageinfo

    def change_device(self):
        while True:
            command = input()
            op = ''
            if command.startswith(':+'):
                device_no = int(command.strip(':+'))
                op = 'add'
            elif command.startswith(':-'):
                device_no = int(command.strip(':-'))
                op = 'remove'
            else:
                print(f'Invalid command {command}')
                continue
            if isinstance(device_no, int) and 0 <= device_no <= 9:
                assert op != ''
                if op == 'add':
                    if device_no not in self.allow_devices:
                        self.allow_devices.append(device_no)
                    else:
                        print(f'Device {device_no} is already in the queue.')
                elif op == 'remove':
                    if device_no in self.allow_devices:
                        self.allow_devices.remove(device_no)
                    else:
                        print(f'Device {device_no} is not in the queue.')
                self.allow_devices.sort()
                print(f'Allowed device: {self.allow_devices}.')
            else:
                print(f'Invalid device number: {device_no}')

