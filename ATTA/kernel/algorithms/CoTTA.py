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
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random
import PIL
from .Base import AlgBase
import pandas as pd

@register.alg_register
class CoTTA(AlgBase):
    def __init__(self, config: Conf):
        super(CoTTA, self).__init__(config)
        print('#D#Config model')
        self.configure_model()
        params, param_names = self.collect_params()
        # print(f'#I#{param_names}')
        self.optimizer = torch.optim.SGD(params, lr=self.config.atta.SAR.lr, momentum=0.9)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self.copy_model_and_optimizer()
        self.transform = get_tta_transforms()


    def __call__(self, *args, **kwargs):
        # super(CoTTA, self).__call__()

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

    @staticmethod
    def softmax_entropy(x, x_ema):  # -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

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
                outputs = self.forward_and_adapt(data, targets, gt_mask)
            acc += self.config.metric.score_func(targets, outputs) * len(data)
        acc /= len(loader[env_id].sampler)
        print(f'Env {env_id} real-time Acc.: {acc:.4f}')
        return acc

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        model_anchor = deepcopy(self.model)
        optimizer_state = deepcopy(self.optimizer.state_dict())
        ema_model = deepcopy(self.model)
        for param in ema_model.parameters():
            param.detach_()
        return model_state, optimizer_state, ema_model, model_anchor

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        # use this line if you want to reset the teacher model as well. Maybe you also
        # want to del self.model_ema first to save gpu memory.
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self.copy_model_and_optimizer()

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
            else:
                m.requires_grad_(True)

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if True:  # isinstance(m, nn.BatchNorm2d): collect all
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        print(nm, np)
        return params, names


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, data, targets, gt_mask):
        outputs = self.model(data)
        self.model_ema.train()
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(data), dim=1).max(1)[0]
        standard_ema = self.model_ema(data)
        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        to_aug = anchor_prob.mean(0) < 0.1
        if to_aug:
            for i in range(N):
                outputs_  = self.model_ema(self.transform(data)).detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Augmentation-averaged Prediction
        # Student update
        loss = (self.softmax_entropy(outputs, outputs_ema.detach())).mean(0)

        # --- AL learning ---
        if self.config.atta.al_rate is not None and gt_mask.sum() > 0:
            loss += self.config.metric.loss_func(outputs[gt_mask], targets[gt_mask])


        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=0.999)
        # Stochastic restore
        # print('Stochastic restore')
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.001).float().to(self.config.device)
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema

def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0, gaussian_std),
        Clip(clip_min, clip_max)
    ])
    return tta_transforms

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.
        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string