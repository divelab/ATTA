from typing import Dict
from typing import Union

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
from torch.utils.data import DataLoader

from ATTA.utils.config_reader import Conf
from ATTA.utils.register import register
from copy import deepcopy
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
# import models for resnet18
from torchvision.models import resnet18
import itertools
import os
import ATTA.data.loaders.misc as misc
from ATTA import register
from ATTA.utils.config_reader import Conf
from ATTA.utils.config_reader import Conf
from ATTA.utils.initial import reset_random_seed
from ATTA.utils.initial import reset_random_seed
from ATTA.data.loaders.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset, Subset
from .Base import AlgBase
import pandas as pd


@register.alg_register
class SAR(AlgBase):
    def __init__(self, config: Conf):
        super(SAR, self).__init__(config)
        num_classes = 2 if config.dataset.num_classes == 1 else config.dataset.num_classes
        self.margin_e0, self.reset_constant_em, self.ema = 0.4 * math.log(
            num_classes), config.atta.SAR.reset_constant_em, None

        print('#D#Config model')
        self.configure_model()
        params, param_names = self.collect_params()
        # print(f'#I#{param_names}')
        self.optimizer = SAM(params, torch.optim.SGD, lr=self.config.atta.SAR.lr, momentum=0.9)
        self.model_state, self.optimizer_state = \
            self.copy_model_and_optimizer()

    def __call__(self, *args, **kwargs):
        # super(SAR, self).__call__(*args, **kwargs)

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
            self.random_result_df.loc['Current step', target_split_id] = self.adapt_on_env(self.target_loader,
                                                                                           target_split_id)
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
        steps = self.config.atta.SAR.steps
        self.configure_model()
        acc = 0
        for data, targets in loader[env_id]:
            targets = targets.to(self.config.device)
            gt_mask = torch.rand(targets.shape[0], device=targets.device) < self.config.atta.al_rate
            for _ in range(steps):
                data = data.to(self.config.device)
                outputs, reset_flag = self.forward_and_adapt(data, targets, gt_mask)
                if reset_flag:
                    self.reset()
            acc += self.config.metric.score_func(targets, outputs) * len(data)
        acc /= len(loader[env_id].sampler)
        print(f'Env {env_id} real-time Acc.: {acc:.4f}')
        return acc

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.ema = None

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
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if 'convs.2.nn' in nm or 'norms.2' in nm:
                continue
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    def update_ema(self, new_data):
        if self.ema is None:
            return new_data
        else:
            with torch.no_grad():
                return 0.9 * self.ema + (1 - 0.9) * new_data

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, data, targets, gt_mask):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()
        # forward
        outputs = self.model(data)
        # adapt
        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = self.softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)

        # --- AL learning ---
        if self.config.atta.al_rate is not None and gt_mask.sum() > 0:
            loss += self.config.metric.loss_func(outputs[gt_mask], targets[gt_mask])


        loss.backward()

        self.optimizer.first_step(
            zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = self.softmax_entropy(self.model(data))
        entropys2 = entropys2[filter_ids_1]  # second time forward
        loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(
            entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.ema = self.update_ema(loss_second.item())  # record moving average loss values for model recovery

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)

        # perform model recovery
        reset_flag = False
        if self.ema is not None:
            if self.ema < self.reset_constant_em:
                print(f"ema < {self.reset_constant_em}, now reset the model")
                reset_flag = True

        return outputs, reset_flag


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
