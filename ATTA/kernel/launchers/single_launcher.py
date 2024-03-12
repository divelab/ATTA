import shlex
import subprocess

from tqdm import tqdm

from ATTA import register
from .basic_launcher import Launcher


@register.launcher_register
class SingleLauncher(Launcher):
    def __init__(self):
        super(SingleLauncher, self).__init__()

    def __call__(self, jobs_group, auto_args):
        jobs_group = super(SingleLauncher, self).__call__(jobs_group, auto_args)
        for cmd_args in tqdm(jobs_group):
            subprocess.run(shlex.split(cmd_args) + ['--gpu_idx', f'{auto_args.allow_devices[0]}'], close_fds=True,
                           stdout=open('debug_out.log', 'a'), stderr=open('debug_error.log', 'a'),
                           start_new_session=False)
