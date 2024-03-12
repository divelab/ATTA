import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset
# import models for resnet18

from ATTA.data.loaders.fast_data_loader import InfiniteDataLoader, FastDataLoader
from ATTA.utils.config_reader import Conf
from ATTA.utils.initial import reset_random_seed
from ATTA.utils.register import register


# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

@register.alg_register
class AlgBase:
    def __init__(self, config: Conf):
        super(AlgBase, self).__init__()

        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)

        reset_random_seed(config)
        self.dataset = register.datasets[config.dataset.name](config.dataset.dataset_root, config.dataset.test_envs,
                                                              config)
        config.dataset.dataset_type = 'image'
        config.dataset.input_shape = self.dataset.input_shape
        config.dataset.num_classes = 1 if self.dataset.num_classes == 2 else self.dataset.num_classes
        config.model.model_level = 'image'
        config.metric.set_score_func(self.dataset.metric)
        config.metric.set_loss_func(self.dataset.task)

        self.config = config

        self.inf_loader = [InfiniteDataLoader(env, weights=None, batch_size=self.config.train.train_bs,
                                              num_workers=self.config.num_workers) for env in self.dataset]
        reset_random_seed(config)
        self.train_split = [np.random.choice(len(env), size=int(len(env) * 0.8), replace=False) for env in self.dataset]
        print(self.train_split)
        self.val_split = [np.setdiff1d(np.arange(len(env)), self.train_split[i]) for i, env in enumerate(self.dataset)]
        self.train_loader = [InfiniteDataLoader(env, weights=None, batch_size=self.config.train.train_bs,
                                                num_workers=self.config.num_workers, subset=self.train_split[i]) for
                             i, env in enumerate(self.dataset)]
        self.val_loader = [FastDataLoader(env, weights=None, batch_size=self.config.train.train_bs,
                                          num_workers=self.config.num_workers,
                                          subset=self.val_split[i], sequential=True) for i, env in
                           enumerate(self.dataset)]
        reset_random_seed(config)
        fast_random = [np.random.permutation(len(env)) for env in self.dataset]
        self.fast_loader = [FastDataLoader(env, weights=None,
                                           batch_size=self.config.atta.batch_size,
                                           num_workers=self.config.num_workers,
                                           subset=fast_random[i], sequential=True) for i, env in
                            enumerate(self.dataset)]

        reset_random_seed(config)
        self.target_dataset = ConcatDataset(
            [env for i, env in enumerate(self.dataset) if i in config.dataset.test_envs[1:]])
        len_target = len(self.target_dataset)
        target_choices = np.random.permutation(len_target)
        len_split = len_target // 4
        self.target_splits = [target_choices[i * len_split: (i + 1) * len_split] for i in range(4)]
        self.target_splits[-1] = target_choices[3 * len_split:]
        self.target_loader = [FastDataLoader(self.target_dataset, weights=None,
                                             batch_size=self.config.atta.batch_size,
                                             num_workers=self.config.num_workers, subset=self.target_splits[i],
                                             sequential=True) for i in range(4)]

        self.encoder = register.models[config.model.name](config).to(self.config.device)

        #     self.fc = self.encoder.fc
        #     self.model = nn.Sequential(self.encoder, self.fc).to(self.config.device)
        # else:
        self.fc = nn.Linear(self.encoder.n_outputs, config.dataset.num_classes).to(self.config.device)
        self.model = nn.Sequential(self.encoder, self.fc).to(self.config.device)

        if 'ImageNet' in config.dataset.name or 'CIFAR' in config.dataset.name:
            self.train_on_env(self.config.dataset.test_envs[0], train_only_fc=True, train_or_load='load')
        else:
            self.train_on_env(self.config.dataset.test_envs[0], train_only_fc=False, train_or_load='load')

    def __call__(self, *args, **kwargs):
        for env_id in self.config.dataset.test_envs:
            self.test_on_env(env_id)

    @torch.no_grad()
    def test_on_env(self, env_id):
        # self.encoder.eval()
        # self.fc.eval()
        self.model.eval()
        test_loss = 0
        test_acc = 0
        for data, target in self.fast_loader[env_id]:
            data, target = data.to(self.config.device), target.to(self.config.device)
            output = self.fc(self.encoder(data))
            test_loss += self.config.metric.loss_func(output, target, reduction='sum').item()
            test_acc += self.config.metric.score_func(target, output) * len(data)
        test_loss /= len(self.fast_loader[env_id].dataset)
        test_acc /= len(self.fast_loader[env_id].dataset)
        print(f'#I#Env {env_id} Test set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
        return test_loss, test_acc

    @torch.no_grad()
    def val_on_env(self, env_id):
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for data, target in self.val_loader[env_id]:
            data, target = data.to(self.config.device), target.to(self.config.device)
            output = self.fc(self.encoder(data))
            val_loss += self.config.metric.loss_func(output, target, reduction='sum').item()
            val_acc += self.config.metric.score_func(target, output) * len(data)
        val_loss /= len(self.val_split[env_id])
        val_acc /= len(self.val_split[env_id])
        return val_loss, val_acc

    @torch.enable_grad()
    def train_on_env(self, env_id, train_only_fc=True, train_or_load='train'):
        if train_or_load == 'train' or not os.path.exists(self.config.ckpt_dir + f'/encoder_{env_id}.pth'):
            # if train_only_fc:
            #     self.encoder.train()
            #     self.fc.train()
            #     optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.config.train.lr)
            #     inf_loader = self.extract_pretrained_feat(self.fast_loader[env_id], self.config.train.train_bs)
            #     for batch_idx, (data, target) in enumerate(inf_loader):
            #         optimizer.zero_grad()
            #         data, target = data.to(self.config.device), target.to(self.config.device)
            #         output = self.fc(data)
            #         loss = self.config.metric.loss_func(output, target)
            #         acc = self.config.metric.score_func(target, output)
            #         loss.backward()
            #         optimizer.step()
            #         if batch_idx % self.config.train.log_interval == 0:
            #             print(f'Iteration: {batch_idx} Loss: {loss.item():.4f} Acc: {acc:.4f}')
            #         if batch_idx > self.config.train.max_iters:
            #             break
            # else:
            # best_val_loss = float('inf')
            best_val_acc = 0
            if train_only_fc:
                self.model.eval()
                self.fc.train()
                optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.config.train.lr)
            else:
                self.model.train()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train.lr)
            for batch_idx, (data, target) in enumerate(self.train_loader[env_id]):
                optimizer.zero_grad()
                data, target = data.to(self.config.device), target.to(self.config.device)
                output = self.fc(self.encoder(data))
                loss = self.config.metric.loss_func(output, target)
                acc = self.config.metric.score_func(target, output)
                loss.backward()
                optimizer.step()
                if batch_idx % self.config.train.log_interval == 0:
                    print(f'Iteration: {batch_idx} Loss: {loss.item():.4f} Acc: {acc:.4f}')
                    val_loss, val_acc = self.val_on_env(env_id)
                    if val_acc > best_val_acc:
                        print(f'New best val acc: {val_acc:.4f}')
                        best_val_acc = val_acc
                        torch.save(self.encoder.state_dict(), self.config.ckpt_dir + f'/encoder_{env_id}.pth')
                        torch.save(self.fc.state_dict(), self.config.ckpt_dir + f'/fc_{env_id}.pth')
                    self.model.train()
                if batch_idx > self.config.train.max_iters:
                    break
        else:
            self.encoder.load_state_dict(
                torch.load(self.config.ckpt_dir + f'/encoder_{env_id}.pth', map_location=self.config.device), strict=False)
            self.fc.load_state_dict(
                torch.load(self.config.ckpt_dir + f'/fc_{env_id}.pth', map_location=self.config.device))

    # @torch.no_grad()
    # def extract_pretrained_feat(self, loader, batch_size):
    #     print('Extracting features from the loader')
    #     feats, targets = [], []
    #     for data, target in loader:
    #         data, target = data.to(self.config.device), target.to(self.config.device)
    #         feat = self.encoder(data)
    #         feats.append(feat.cpu())
    #         targets.append(target.cpu())
    #     feats = torch.cat(feats, dim=0)
    #     targets = torch.cat(targets, dim=0)
    #     feat_dataset = TensorDataset(feats, targets)
    #     weights = misc.make_weights_for_balanced_classes(
    #         feat_dataset) if self.config.dataset.class_balanced else None
    #     inf_loader = InfiniteDataLoader(dataset=feat_dataset, weights=weights,
    #                                     batch_size=batch_size, num_workers=self.config.num_workers)
    #     return inf_loader
