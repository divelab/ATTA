import copy
import pathlib
import time
from typing import Union

import numpy as np
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()

# from sklearn.cluster import KMeans
# from ATTA.utils.fast_pytorch_kmeans import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from typing import Literal

from torch import nn
import torch
# import models for resnet18
from munch import Munch
from ATTA import register
from ATTA.utils.config_reader import Conf
from ATTA.data.loaders.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from .Base import AlgBase
import pandas as pd
from ATTA.definitions import STORAGE_DIR



@register.alg_register
class SimATTA(AlgBase):
    def __init__(self, config: Conf):
        super(SimATTA, self).__init__(config)

        self.teacher = copy.deepcopy(self.model.to('cpu'))

        self.model.to(config.device)
        self.teacher.to(config.device)
        self.update_teacher(0)  # copy student to teacher

        self.budgets = 0
        self.anchors = None
        self.source_anchors = None
        self.buffer = []
        self.n_clusters = 10
        self.nc_increase = self.config.atta.SimATTA.nc_increase
        self.source_n_clusters = 100

        self.cold_start = self.config.atta.SimATTA.cold_start

        self.consistency_weight = 0
        self.alpha_teacher = 0
        self.accumulate_weight = True
        self.weighted_entropy: Union[Literal['low', 'high', 'both'], None] = 'both'
        self.aggressive = True
        self.beta = self.config.atta.SimATTA.beta
        self.alpha = 0.2

        self.target_cluster = True if self.config.atta.SimATTA.target_cluster else False
        self.LE = True if self.config.atta.SimATTA.LE else False
        self.vis_round = 0


    def __call__(self, *args, **kwargs):
        # super(SimATTA, self).__call__()
        self.continue_result_df = pd.DataFrame(
            index=['Current domain', 'Budgets', *(i for i in self.config.dataset.test_envs), 'Frame AVG'],
            columns=[*(i for i in self.config.dataset.test_envs), 'Test AVG'], dtype=float)
        self.random_result_df = pd.DataFrame(
            index=['Current step', 'Budgets', *(i for i in self.config.dataset.test_envs), 'Frame AVG'],
            columns=[*(i for i in range(4)), 'Test AVG'], dtype=float)

        self.enable_bn(self.model)
        if 'ImageNet' not in self.config.dataset.name:
            for env_id in self.config.dataset.test_envs:
                acc = self.test_on_env(env_id)[1]
                self.continue_result_df.loc[env_id, self.config.dataset.test_envs[0]] = acc
                self.random_result_df.loc[env_id, self.config.dataset.test_envs[0]] = acc

        for adapt_id in self.config.dataset.test_envs[1:]:
            self.continue_result_df.loc['Current domain', adapt_id] = self.adapt_on_env(self.fast_loader, adapt_id)
            self.continue_result_df.loc['Budgets', adapt_id] = self.budgets
            print(self.budgets)
            if 'ImageNet' not in self.config.dataset.name:
                for env_id in self.config.dataset.test_envs:
                    self.continue_result_df.loc[env_id, adapt_id] = self.test_on_env(env_id)[1]

        self.__init__(self.config)
        for target_split_id in range(4):
            self.random_result_df.loc['Current step', target_split_id] = self.adapt_on_env(self.target_loader, target_split_id)
            self.random_result_df.loc['Budgets', target_split_id] = self.budgets
            print(self.budgets)
            if 'ImageNet' not in self.config.dataset.name:
                for env_id in self.config.dataset.test_envs:
                    self.random_result_df.loc[env_id, target_split_id] = self.test_on_env(env_id)[1]

        print(f'#IM#\n{self.continue_result_df.round(4).to_markdown()}\n'
              f'{self.random_result_df.round(4).to_markdown()}')
        # print(self.random_result_df.round(4).to_markdown(), '\n')
        self.continue_result_df.round(4).to_csv(f'{self.config.log_file}.csv')
        self.random_result_df.round(4).to_csv(f'{self.config.log_file}.csv', mode='a')


    @torch.no_grad()
    def val_anchor(self, loader):
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for data, target in loader:
            data, target = data.to(self.config.device), target.to(self.config.device)
            output = self.fc(self.encoder(data))
            val_loss += self.config.metric.loss_func(output, target, reduction='sum').item()
            val_acc += self.config.metric.score_func(target, output) * len(data)
        val_loss /= len(loader.sampler)
        val_acc /= len(loader.sampler)
        return val_loss, val_acc

    def update_teacher(self, alpha_teacher):  # , iteration):
        for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
            t_param.data[:] = alpha_teacher * t_param[:].data[:] + (1 - alpha_teacher) * s_param[:].data[:]
        if not self.config.model.freeze_bn:
            for tm, m in zip(self.teacher.modules(), self.model.modules()):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    tm.running_mean = alpha_teacher * tm.running_mean + (1 - alpha_teacher) * m.running_mean
                    tm.running_var = alpha_teacher * tm.running_var + (1 - alpha_teacher) * m.running_var

    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors):
        self.model.train()

        source_loader = InfiniteDataLoader(TensorDataset(source_anchors.data, source_anchors.target), weights=None,
                                           batch_size=self.config.train.train_bs,
                                           num_workers=self.config.num_workers)
        target_loader = InfiniteDataLoader(TensorDataset(target_anchors.data, target_anchors.target), weights=None,
                                             batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)
        alpha = target_anchors.num_elem() / (target_anchors.num_elem() + source_anchors.num_elem())
        if source_anchors.num_elem() < self.cold_start:
            alpha = min(0.2, alpha)

        ST_loader = iter(zip(source_loader, target_loader))
        val_loader = FastDataLoader(TensorDataset(target_anchors.data, target_anchors.target), weights=None,
                                    batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
        # print('Cluster train')
        delay_break = False
        loss_window = []
        tol = 0
        lowest_loss = float('inf')
        for i, ((S_data, S_targets), (T_data, T_targets)) in enumerate(ST_loader):
            S_data, S_targets = S_data.to(self.config.device), S_targets.to(self.config.device)
            T_data, T_targets = T_data.to(self.config.device), T_targets.to(self.config.device)
            L_T = self.one_step_train(S_data, S_targets, T_data, T_targets, alpha, optimizer)
            # self.update_teacher(self.alpha_teacher)
            if len(loss_window) < self.config.atta.SimATTA.stop_tol:
                loss_window.append(L_T.item())
            else:
                mean_loss = np.mean(loss_window)
                tol += 1
                if mean_loss < lowest_loss:
                    lowest_loss = mean_loss
                    tol = 0
                if tol > 5:
                    break
                loss_window = []
            if 'ImageNet' in self.config.dataset.name or 'CIFAR' in self.config.dataset.name:
                if i > self.config.atta.SimATTA.steps:
                    break


    def one_step_train(self, S_data, S_targets, T_data, T_targets, alpha, optimizer):
        # print('one step train')
        L_S = self.config.metric.loss_func(self.model(S_data), S_targets)
        L_T = self.config.metric.loss_func(self.model(T_data), T_targets)
        loss = (1 - alpha) * L_S + alpha * L_T
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return L_T

    def softmax_entropy(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        if y is None:
            if x.shape[1] == 1:
                x = torch.cat([x, -x], dim=1)
            return -(x.softmax(1) * x.log_softmax(1)).sum(1)
        else:
            return - 0.5 * (x.softmax(1) * y.log_softmax(1)).sum(1) - 0.5 * (y.softmax(1) * x.log_softmax(1)).sum(1)

    def update_anchors(self, anchors, data, target, feats, weight):
        if anchors is None:
            anchors = Munch()
            anchors.data = data
            anchors.target = target
            anchors.feats = feats
            anchors.weight = weight
            anchors.num_elem = lambda: len(anchors.data)
        else:
            anchors.data = torch.cat([anchors.data, data])
            anchors.target = torch.cat([anchors.target, target])
            anchors.feats = torch.cat([anchors.feats, feats])
            anchors.weight = torch.cat([anchors.weight, weight])
        return anchors

    def update_anchors_feats(self, anchors):
        # sequential_data = torch.arange(200)[:, None]
        anchors_loader = FastDataLoader(TensorDataset(anchors.data), weights=None,
                                        batch_size=32, num_workers=self.config.num_workers, sequential=True)

        anchors.feats = None
        self.model.eval()
        for data in anchors_loader:
            # print(data)
            data = data[0].to(self.config.device)
            if anchors.feats is None:
                anchors.feats = self.model[0](data).cpu().detach()
            else:
                anchors.feats = torch.cat([anchors.feats, self.model[0](data).cpu().detach()])

        return anchors

    @torch.no_grad()
    def adapt_on_env(self, loader, env_id):
        # beta_func = torch.distributions.beta.Beta(0.8, 0.8)
        acc = 0
        for data, target in tqdm(loader[env_id]):
            data, target = data.to(self.config.device), target.to(self.config.device)
            outputs, closest, self.anchors = self.sample_select(self.model, data, target, self.anchors, int(self.n_clusters), 1, ent_bound=self.config.atta.SimATTA.eh, incremental_cluster=self.target_cluster)
            acc += self.config.metric.score_func(target, outputs).item() * data.shape[0]
            if self.LE:
                _, _, self.source_anchors = self.sample_select(self.teacher, data, target, self.source_anchors, self.source_n_clusters, 0,
                                                               use_pseudo_label=True, ent_bound=self.config.atta.SimATTA.el, incremental_cluster=False)
            else:
                self.source_anchors = self.update_anchors(None, torch.tensor([]), None, None, None)
            if not self.target_cluster:
                self.n_clusters = 0
            self.source_n_clusters = 100

            self.budgets += len(closest)
            self.n_clusters += self.nc_increase
            self.source_n_clusters += 1

            print(self.anchors.num_elem(), self.source_anchors.num_elem())
            if self.source_anchors.num_elem() > 0:
                self.cluster_train(self.anchors, self.source_anchors)
            else:
                self.cluster_train(self.anchors, self.anchors)
            self.anchors = self.update_anchors_feats(self.anchors)
        acc /= len(loader[env_id].sampler)
        print(f'#IN#Env {env_id} real-time Acc.: {acc:.4f}')
        return acc

    @torch.no_grad()
    def sample_select(self, model, data, target, anchors, n_clusters, ent_beta, use_pseudo_label=False, ent_bound=1e-2, incremental_cluster=False):
        model.eval()
        feats = model[0](data)
        outputs = model[1](feats)
        pseudo_label = outputs.argmax(1).cpu().detach()
        data = data.cpu().detach()
        feats = feats.cpu().detach()
        target = target.cpu().detach()
        entropy = self.softmax_entropy(outputs).cpu()
        if not incremental_cluster:
            entropy = entropy.numpy()
            if ent_beta == 0:
                closest = np.argsort(entropy)[: n_clusters]
                closest = closest[entropy[closest] < ent_bound]
            elif ent_beta == 1:
                closest = np.argsort(entropy)[- n_clusters:]
                closest = closest[entropy[closest] >= ent_bound]
            else:
                raise NotImplementedError
            weights = torch.zeros(len(closest), dtype=torch.float)
        else:
            if ent_beta == 0:
                sample_choice = entropy < ent_bound
            elif ent_beta == 1:
                sample_choice = entropy >= ent_bound
            else:
                raise NotImplementedError

            data = data[sample_choice]
            target = target[sample_choice]
            feats = feats[sample_choice]
            pseudo_label = pseudo_label[sample_choice]

            if anchors:
                feats4cluster = torch.cat([anchors.feats, feats])
                sample_weight = torch.cat([anchors.weight, torch.ones(len(feats), dtype=torch.float)])
            else:
                feats4cluster = feats
                sample_weight = torch.ones(len(feats), dtype=torch.float)

            if self.config.atta.gpu_clustering:
                from ATTA.utils.fast_pytorch_kmeans import KMeans
                from joblib import parallel_backend
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, device=self.config.device).fit(
                    feats4cluster.to(self.config.device),
                    sample_weight=sample_weight.to(self.config.device))
                with parallel_backend('threading', n_jobs=8):
                    raw_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats4cluster)
                kmeans_labels = kmeans.labels_
            # elif self.config.atta.gpu_clustering == 'jax':
            #     from ott.tools.k_means import k_means as KMeans
            #     import jax
            #     import jax.numpy as jnp
            #     tik = time.time()
            #     kmeans = KMeans(jnp.array(feats4cluster.numpy()), k=n_clusters, weights=jnp.array(sample_weight.numpy()), n_init=10)
            #     mit = time.time()
            #     print(f'#IN#Kmeans time: {mit - tik}')
            #     @jax.jit
            #     def jax_pairwise_distances_argmin(c, feats):
            #         dis = lambda x, y: jnp.sqrt(((x - y) ** 2).sum())
            #         argmin_dis = lambda x, y: jnp.argmin(jax.vmap(dis, in_axes=(None, 0))(x, y))
            #         return jax.vmap(argmin_dis, in_axes=(0, None))(c, feats)
            #     raw_closest = np.array(jax_pairwise_distances_argmin(kmeans.centroids, jnp.array(feats4cluster.numpy())))
            #     print(f'#IN#Pairwise distance time: {time.time() - mit}')
            #     kmeans_labels = np.array(kmeans.assignment)
            else:
                from joblib import parallel_backend
                from sklearn.cluster import KMeans
                with parallel_backend('threading', n_jobs=8):
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, algorithm='elkan').fit(feats4cluster,
                                                                                                  sample_weight=sample_weight)
                    raw_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats4cluster)
                kmeans_labels = kmeans.labels_



            if anchors:
                num_anchors = anchors.num_elem()
                prev_anchor_cluster = torch.tensor(kmeans_labels[:num_anchors], dtype=torch.long)

                if self.accumulate_weight:
                    # previous anchor weight accumulation
                    # Average the weight of the previous anchor if sharing the same cluster
                    num_prev_anchors_per_cluster = prev_anchor_cluster.unique(return_counts=True)
                    num_prev_anchors_per_cluster_dict = torch.zeros(len(raw_closest), dtype=torch.long)
                    num_prev_anchors_per_cluster_dict[num_prev_anchors_per_cluster[0].long()] = \
                    num_prev_anchors_per_cluster[1]

                    num_newsample_per_cluster = torch.tensor(kmeans_labels).unique(return_counts=True)
                    num_newsample_per_cluster_dict = torch.zeros(len(raw_closest), dtype=torch.long)
                    num_newsample_per_cluster_dict[num_newsample_per_cluster[0].long()] = num_newsample_per_cluster[1]
                    assert (num_prev_anchors_per_cluster_dict[prev_anchor_cluster] == 0).sum() == 0
                    # accumulate the weight of the previous anchor
                    anchors.weight = anchors.weight + num_newsample_per_cluster_dict[prev_anchor_cluster] / \
                                          num_prev_anchors_per_cluster_dict[prev_anchor_cluster].float()

                anchored_cluster_mask = torch.zeros(len(raw_closest), dtype=torch.bool).index_fill_(0,
                                                                                                    prev_anchor_cluster.unique().long(),
                                                                                                    True)
                new_cluster_mask = ~ anchored_cluster_mask

                closest = raw_closest[new_cluster_mask] - num_anchors
                if (closest < 0).sum() != 0:
                    # The cluster's closest sample may not belong to the cluster. It makes sense to eliminate them.
                    print('new_cluster_mask: ', new_cluster_mask)
                    new_cluster_mask = torch.where(new_cluster_mask)[0]
                    print('new_cluster_mask: ', new_cluster_mask)
                    print(closest)
                    print(closest >= 0)
                    new_cluster_mask = new_cluster_mask[closest >= 0]
                    closest = closest[closest >= 0]


                weights = torch.tensor(kmeans_labels).unique(return_counts=True)[1][new_cluster_mask]
            else:
                num_anchors = 0
                closest = raw_closest
                weights = torch.tensor(kmeans_labels).unique(return_counts=True)[1]

        if use_pseudo_label:
            anchors = self.update_anchors(anchors, data[closest], pseudo_label[closest], feats[closest], weights)
        else:
            anchors = self.update_anchors(anchors, data[closest], target[closest], feats[closest], weights)

        return outputs, closest, anchors

    def enable_bn(self, model):
        if not self.config.model.freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = 0.1


