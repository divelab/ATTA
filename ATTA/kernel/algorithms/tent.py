from torch import nn
import torch
# import models for resnet18
from ATTA import register
from ATTA.utils.config_reader import Conf
from .Base import AlgBase
import pandas as pd

@register.alg_register
class Tent(AlgBase):
    def __init__(self, config: Conf):
        super(Tent, self).__init__(config)
        print('#D#Config model')
        self.configure_model()
        params, param_names = self.collect_params()
        print(f'#I#{param_names}')
        self.optimizer = torch.optim.SGD(params, lr=self.config.atta.Tent.lr, momentum=0.9)

    def __call__(self, *args, **kwargs):
        # super(Tent, self).__call__(*args, **kwargs)

        self.continue_result_df = pd.DataFrame(
            index=['Current domain', 'Budgets', *(i for i in self.config.dataset.test_envs), 'Frame AVG'],
            columns=[*(i for i in self.config.dataset.test_envs), 'Test AVG'], dtype=float)
        self.random_result_df = pd.DataFrame(
            index=['Current step', 'Budgets', *(i for i in self.config.dataset.test_envs), 'Frame AVG'],
            columns=[*(i for i in range(4)), 'Test AVG'], dtype=float)

        for adapt_id in self.config.dataset.test_envs[1:]:
            self.continue_result_df.loc['Current domain', adapt_id] = self.adapt_on_env(self.fast_loader, adapt_id)
            # for env_id in self.config.dataset.test_envs:
            #     self.continue_result_df.loc[env_id, adapt_id] = self.test_on_env(env_id)[1]

        self.__init__(self.config)
        for target_split_id in range(4):
            self.random_result_df.loc['Current step', target_split_id] = self.adapt_on_env(self.target_loader, target_split_id)
            # for env_id in self.config.dataset.test_envs:
            #     self.random_result_df.loc[env_id, target_split_id] = self.test_on_env(env_id)[1]

        print(self.continue_result_df.round(4).to_markdown(), '\n')
        print(self.random_result_df.round(4).to_markdown())




    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        if x.shape[1] == 1:
            x = torch.cat([x, -x], dim=1)
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    # Active contrastive learning
    @torch.enable_grad()
    def adapt_on_env(self, loader, env_id):
        steps = self.config.atta.Tent.steps
        self.configure_model()
        acc = 0
        for data, targets in loader[env_id]:
            targets = targets.to(self.config.device)
            if self.config.atta.al_rate is not None:
                gt_mask = torch.rand(targets.shape[0], device=targets.device) < self.config.atta.al_rate
            for _ in range(steps):
                data = data.to(self.config.device)
                outputs = self.model(data)
                loss = self.softmax_entropy(outputs).mean(0)

                # --- AL learning ---
                if self.config.atta.al_rate is not None and gt_mask.sum() > 0:
                    loss += self.config.metric.loss_func(outputs[gt_mask], targets[gt_mask])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # for env_id in self.config.dataset.test_envs:
                #     self.test_on_env(env_id)
            acc += self.config.metric.score_func(targets, outputs) * len(data)
        acc /= len(loader[env_id].sampler)
        print(f'Env {env_id} real-time Acc.: {acc:.4f}')
        return acc

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
