r"""An important module that is used to define all arguments for both argument container and configuration container.
"""
import pathlib
import sys
from typing import List

from tap import Tap
from typing_extensions import Literal

from ATTA.definitions import ROOT_DIR


class TreeTap(Tap):

    def __init__(self, argv=None, tree_parser=None):
        super(TreeTap, self).__init__()
        self.skipped_args = []
        if tree_parser:
            self.argv = []
            skip_arg = False
            for arg in argv:
                if arg.startswith('--'):
                    skip_arg = True
                    if arg.startswith(f'--{tree_parser}.'):
                        self.argv.append(f'--{".".join(arg.split(".")[1:])}')
                        skip_arg = False
                elif not skip_arg:
                    self.argv.append(arg)
                if skip_arg:
                    self.skipped_args.append(arg)
        else:
            self.argv = sys.argv[1:] if argv is None else argv

    def parse_args(self):
        return super(TreeTap, self).parse_args(self.argv, known_only=True)

    def process_args(self) -> None:
        super(TreeTap, self).process_args()
        for action in self._actions:
            if isinstance(action.type, type) and issubclass(action.type, Tap):
                setattr(self, action.dest, action.type(self.extra_args, tree_parser=action.dest).parse_args())

                # Remove parsed arguments
                self.extra_args = getattr(self, action.dest).skipped_args
        if self.extra_args:
            extra_keys = [arg[2:] for arg in self.extra_args if arg.startswith('--')]
            raise ValueError(f"Unexpected arguments [{', '.join(extra_keys)}] in {self.__class__.__name__}")

class TrainArgs(TreeTap):
    r"""
    Correspond to ``train`` configs in config files.
    """
    tr_ctn: bool = None  #: Flag for training continue.
    ctn_epoch: int = None  #: Start epoch for continue training.
    max_epoch: int = None  #: Max epochs for training stop.
    save_gap: int = None  #: Hard checkpoint saving gap.
    pre_train: int = None  #: Pre-train epoch before picking checkpoints.
    log_interval: int = None  #: Logging interval.
    max_iters: int = None  #: Max iterations for training stop.

    train_bs: int = None  #: Batch size for training.
    val_bs: int = None  #: Batch size for validation.
    test_bs: int = None  #: Batch size for test.
    num_steps: int = None  #: Number of steps in each epoch for node classifications.

    lr: float = None  #: Learning rate.
    epoch: int = None  #: Current training epoch. This value should not be set manually.
    stage_stones: List[int] = None  #: The epoch for starting the next training stage.
    mile_stones: List[int] = None  #: Milestones for a scheduler to decrease learning rate: 0.1
    weight_decay: float = None  #: Weight decay.
    gamma: float = None  #: Gamma for a scheduler to decrease learning rate: 0.1

    alpha = None  #: A parameter for DANN.


class DatasetArgs(TreeTap):
    r"""
    Correspond to ``dataset`` configs in config files.
    """
    name: str = None  #: Name of the chosen dataset.
    dataloader_name: str = None#: Name of the chosen dataloader. The default is BaseDataLoader.
    shift_type: Literal['no_shift', 'covariate', 'concept'] = None  #: The shift type of the chosen dataset.
    domain: str = None  #: Domain selection.
    generate: bool = None  #: The flag for generating ATTA datasets from scratch instead of downloading
    dataset_root: str = None  #: Dataset storage root. Default STORAGE_ROOT/datasets
    dataset_type: str = None  #: Dataset type: molecule, real-world, synthetic, etc. For special usages.
    class_balanced: bool = None  #: Whether to use class balanced sampler.
    data_augmentation: bool = None  #: Whether to use data augmentation.


    dim_node: int = None  #: Dimension of node
    dim_edge: int = None  #: Dimension of edge
    num_classes: int = None  #: Number of labels for multi-label classifications.
    num_envs: int = None  #: Number of environments in training set.
    num_domains: int = None  #: Number of domains in training set.
    feat_dims: List[int] = None  #: Number of integer values for each x feature.
    edge_feat_dims: List[int] = None  #: Number of integer values for each edge feature.
    test_envs: List[int] = None  #: Test environments.


class ModelArgs(TreeTap):
    r"""
    Correspond to ``model`` configs in config files.
    """
    name: str = None  #: Name of the chosen GNN.
    model_layer: int = None  #: Number of the GNN layer.
    model_level: Literal['node', 'link', 'graph', 'image'] = 'graph'  #: What is the model use for? Node, link, or graph predictions.
    nonlinear_classifier: bool = None  #: Whether to use a nonlinear classifier.
    resnet18: bool = None  #: Whether to use a ResNet18 backbone.

    dim_hidden: int = None  #: Node hidden feature's dimension.
    dim_ffn: int = None  #: Final linear layer dimension.
    global_pool: str = None  #: Readout pooling layer type. Currently allowed: 'max', 'mean'.
    dropout_rate: float = None  #: Dropout rate.
    freeze_bn: bool = None  #: Whether to freeze batch normalization layers.


class OODArgs(TreeTap):
    r"""
    Correspond to ``ood`` configs in config files.
    """
    alg: str = None  #: Name of the chosen OOD algorithm.
    ood_param: float = None  #: OOD algorithms' hyperparameter(s). Currently, most of algorithms use it as a float value.
    extra_param: List = None  #: OOD algorithms' extra hyperparameter(s).

    def process_args(self) -> None:
        self.extra_param = [eval(param) for param in self.extra_param] if self.extra_param is not None else None

class SARArgs(TreeTap):
    steps: int = None
    reset_constant_em: float = None
    lr: float = None

class TentArgs(TreeTap):
    steps: int = None
    lr: float = None

class EATAArgs(TreeTap):
    steps: int = None
    lr: float = None
    d_margin: float = None
    fisher_alpha: float = None

class TICPArgs(TreeTap):
    num_drops: int = None #: Number of pseudo environments.
    classifier_epochs: int = None #: Number of epochs for training the classifiers
    classifier_lr: float = None #: Learning rate for training the classifiers
    num_gaussians: int = None #: Number of Gaussian distributions for the mixture model
    alpha: float = None #: Alpha parameter for the mixture model
    beta: float = None #: Beta parameter for the mixture model
    test_interval: int = None #: Interval for testing the classifiers
    train_linear_classifier: bool = None #: Whether to train a linear classifier
    steps: int = None #: Number of steps for the TICP algorithm
    temperature: float = None #: Temperature for the TICP algorithm
    train_or_load: str = None #: Whether to train or load the classifiers/vaes

class TTAArgs(TreeTap):
    name: str = None #: Name of the TTA algorithm
    episodic: bool = None
    SAR: SARArgs = None
    Tent: TentArgs = None
    TICP: TICPArgs = None

class TALArgs(TreeTap):
    steps: int = None
    lr: float = None
    e0: float = None


class SimATTAArgs(TreeTap):
    steps: int = None #: Number of steps for the ATTA algorithm
    lr: float = None  #: Learning rate for the ATTA algorithm
    eh: float = None #: Initial entropy high threshold
    el: float = None #: Initial entropy low threshold
    cold_start: int = None #: Number of steps for the cold start phase
    beta: float = None #: Beta parameter for the ATTA algorithm
    nc_increase: float = None #: Number of clusters to increase for each time step
    stop_tol: int = None #: Stopping tolerance for the ATTA algorithm
    target_cluster: int = None #: Whether to use target clustering for the ATTA algorithm
    LE: int = None #: Whether to use low-entropy samples for the ATTA algorithm

class ATTAArgs(TreeTap):
    name: str = None #: Name of the ATTA algorithm
    budgets: int = None #: Number of samples to be labeled
    episodic: bool = None
    batch_size: int = None #: Batch size for the ATTA algorithms
    al_rate: float = None #: Whether to use active learning for the ATTA algorithm
    gpu_clustering: bool = None #: GPU flag for the torch clustering implementation.
    SimATTA: SimATTAArgs = None
    Tent: TentArgs = None
    TAL: TALArgs = None
    SAR: SARArgs = None
    EATA: EATAArgs = None


class AutoArgs(Tap):
    config_root: str = None  #: The root of input configuration files.
    sweep_root: str = None  #: The root of hyperparameter searching configurations.
    final_root: str = None  #: The root of output final configuration files.
    launcher: str = None  #: The launcher name.


    allow_datasets: List[str] = None  #: Allow datasets in list to run.
    allow_domains: List[str] = None  #: Allow domains in list to run.
    allow_shifts: List[str] = None  #: Allow shifts.
    allow_algs: List[str] = None  #: Allowed OOD algorithms.
    allow_devices: List[int] = None  #: Devices allowed to run.
    allow_rounds: List[int] = None # The numbers of experiment round.


class CommonArgs(TreeTap):
    r"""
    Correspond to general configs in config files.
    """
    config_path: pathlib.Path  #: (Required) The path for the config file.

    task: Literal['train', 'test', 'adapt'] = None  #: Running mode. Allowed: 'train' and 'test'.
    random_seed: int = None  #: Fixed random seed for reproducibility.
    exp_round: int = None  #: Current experiment round.
    pytest: bool = None
    pipeline: str = None  #: Training/test controller.

    ckpt_root: str = None  #: Checkpoint root for saving checkpoint files, where inner structure is automatically generated
    ckpt_dir: str = None  #: The direct directory for saving ckpt files
    test_ckpt: str = None  #: Path of the model general test or out-of-domain test checkpoint
    id_test_ckpt: str = None  #: Path of the model in-domain checkpoint
    save_tag: str = None  #: Special save tag for distinguishing special training checkpoints.
    other_saved = None  #: Other info that need to be stored in a checkpoint.
    clean_save: bool = None  #: Only save necessary checkpoints.
    full_clean: bool = None

    gpu_idx: int = None  #: GPU index.
    device = None  #: Automatically generated by choosing gpu_idx.
    num_workers: int = None  #: Number of workers used by data loaders.

    log_file: str = None  #: Log file name.
    log_path: str = None  #: Log file path.

    tensorboard_logdir: str = None  #: Tensorboard logging place.

    mp_spawn: bool = None  #: Whether to use multiprocessing spawn method.

    # For code auto-complete
    train: TrainArgs = None  #: For code auto-complete
    model: ModelArgs = None  #: For code auto-complete
    dataset: DatasetArgs = None  #: For code auto-complete
    ood: OODArgs = None  #: For code auto-complete
    tta: TTAArgs = None
    atta: ATTAArgs = None

    def __init__(self, argv):
        super(CommonArgs, self).__init__(argv)

        from ATTA.utils.metric import Metric
        self.metric: Metric = None

    def process_args(self) -> None:
        super().process_args()
        if not self.config_path.is_absolute():
            self.config_path = pathlib.Path(ROOT_DIR) / 'configs' / self.config_path


def args_parser(argv: list=None):
    r"""
    Arguments parser.

    Args:
        argv: Input arguments. *e.g.*, ['--config_path', config_path,
            '--ckpt_root', os.path.join(STORAGE_DIR, 'reproduce_ckpts'),
            '--exp_round', '1']

    Returns:
        General arguments

    """
    common_args = CommonArgs(argv=argv).parse_args()
    return common_args
