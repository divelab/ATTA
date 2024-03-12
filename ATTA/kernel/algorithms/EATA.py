# import models for resnet18
import os
from copy import deepcopy

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import ATTA.data.loaders.misc as misc
from ATTA import register
from ATTA.data.loaders.fast_data_loader import InfiniteDataLoader, FastDataLoader
from ATTA.utils.config_reader import Conf
from ATTA.utils.initial import reset_random_seed
import torch.nn.functional as F
from .Base import AlgBase
import pandas as pd


@register.alg_register
class EATA(AlgBase):

    def __init__(self, config: Conf):
        super(EATA, self).__init__(config)
        num_classes = 2 if config.dataset.num_classes == 1 else config.dataset.num_classes

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = 0.4 * math.log(num_classes)  # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = self.config.atta.EATA.d_margin  # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)

        self.fishers = None  # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = self.config.atta.EATA.fisher_alpha  # trade-off \beta for two losses (Eqn. 8)

        print('load fisher loader')
        fisher_loader = DataLoader(self.target_dataset, sampler=torch.utils.data.SubsetRandomSampler(
            np.random.choice(len(self.target_dataset), size=2000, replace=False)), batch_size=64,
                                   num_workers=self.config.num_workers)

        self.configure_model()
        params, param_names = self.collect_params()
        # fishers = None
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().to(self.config.device)
        print('train fisher')
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):
            print(iter_)
            images, targets = images.to(self.config.device), targets.to(self.config.device)
            outputs = self.model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        print("compute fisher matrices finished")
        del ewc_optimizer

        self.fishers = fishers

        self.optimizer = torch.optim.SGD(params, self.config.atta.EATA.lr, momentum=0.9)
        self.model_state, self.optimizer_state = \
            self.copy_model_and_optimizer()


    def __call__(self, *args, **kwargs):
        # super(EATA, self).__call__()

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
        temprature = 1
        x = x / temprature
        x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
        return x

    # Active contrastive learning
    @torch.enable_grad()
    def adapt_on_env(self, loader, env_id):
        steps = self.config.atta.EATA.steps
        self.configure_model()
        acc = 0
        for data, targets in loader[env_id]:
            targets = targets.to(self.config.device)
            gt_mask = torch.rand(targets.shape[0], device=targets.device) < self.config.atta.al_rate
            for _ in range(steps):
                data = data.to(self.config.device)
                outputs, num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt(data, targets, gt_mask)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
            acc += self.config.metric.score_func(targets, outputs) * len(data)
        acc /= len(loader[env_id].sampler)
        print(f'Env {env_id} real-time Acc.: {acc:.4f}')
        return acc

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

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
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, data, targets, gt_mask):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return:
        1. model outputs;
        2. the number of reliable and non-redundant samples;
        3. the number of reliable samples;
        4. the moving average  probability vector over all previous samples
        """
        # forward
        outputs = self.model(data)
        # adapt
        entropys = self.softmax_entropy(outputs)
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0]>-0.1)
        entropys = entropys[filter_ids_1]
        # filter redundant samples
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = self.update_model_probs(outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = self.update_model_probs(outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # loss = 0
        # if x[ids1][ids2].size(0) != 0:
        #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1])**2).sum()
            loss += ewc_loss

        # --- AL learning ---
        if self.config.atta.al_rate is not None and gt_mask.sum() > 0:
            loss += self.config.metric.loss_func(outputs[gt_mask], targets[gt_mask])

        if data[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs

    def update_model_probs(self, new_probs):
        if self.current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return self.current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * self.current_model_probs + (1 - 0.9) * new_probs.mean(0)



class ActualSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)
