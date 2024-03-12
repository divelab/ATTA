import numpy as np
import torch
from joblib import parallel_backend
from sklearn.cluster import KMeans
# from ATTA.utils.fast_pytorch_kmeans import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import ConcatDataset
# import models for resnet18

from ATTA.data.loaders.fast_data_loader import InfiniteDataLoader
from ATTA.utils.config_reader import Conf
from ATTA.utils.register import register
from .Base import AlgBase

@register.alg_register
class Kmeans(AlgBase):
    def __init__(self, config: Conf):
        super(Kmeans, self).__init__(config)
        self.budgets = self.config.atta.budgets
        self.anchors = []
        self.buffer = []


    def __call__(self, *args, **kwargs):
        self.adapt()
        for env_id in self.config.dataset.test_envs:
            self.test_on_env(env_id)

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        if x.shape[1] == 1:
            x = torch.cat([x, -x], dim=1)
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    @torch.no_grad()
    def adapt(self):
        idxs_lb = np.zeros(len(self.target_dataset), dtype=bool)
        for round in range(10):
            data_loader = torch.utils.data.DataLoader(self.target_dataset, sampler=ActualSequentialSampler(np.arange(len(self.target_dataset))),
                                                      num_workers=self.config.num_workers,
                                                      batch_size=self.config.train.train_bs, drop_last=False)
            self.model.eval()
            buffer = []
            for data, target in data_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                feats = self.encoder(data)
                output = self.fc(feats)
                entropy = self.softmax_entropy(output)
                buffer.append((data, target, feats.cpu().numpy(), entropy.cpu().numpy()))
            data, target, feats, entropy = zip(*buffer)
            feats = np.concatenate(feats)[~idxs_lb]
            # entropy = np.concatenate(entropy)[~idxs_lb]
            if round == 9:
                n_clusters = self.budgets - idxs_lb.sum()
            else:
                n_clusters = self.budgets // 10
            with parallel_backend('threading', n_jobs=8):
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, algorithm='elkan').fit(feats)
                closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats)
            idxs_lb_id = idxs_lb.nonzero()[0]
            # print(f'closest: {closest}')
            for idx in closest:
                true_idx = idx
                while true_idx != (idxs_lb_id <= true_idx).sum() + idx:
                    true_idx += 1
                # assert not idxs_lb[true_idx], f'true_idx: {true_idx}, (idxs_lb_id < true_idx).sum(): {(idxs_lb_id < true_idx).sum()}, idx: {idx}'
                idxs_lb[true_idx] = True
            # print(f'idxs_lb: {idxs_lb.nonzero()[0]}')

            anchor_loader = InfiniteDataLoader(self.target_dataset, weights=None,
                                               batch_size=self.config.train.train_bs,
                                               num_workers=self.config.num_workers, subset=np.nonzero(idxs_lb)[0])
            self.model.train()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
            print('Cluster train')
            with torch.enable_grad():
                for i, (data, target) in enumerate(anchor_loader):
                    data, target = data.to(self.config.device), target.to(self.config.device)
                    optimizer.zero_grad()
                    output = self.fc(self.encoder(data))
                    loss = self.config.metric.loss_func(output, target)
                    loss.backward()
                    optimizer.step()
                    if i > self.config.atta.SimATTA.steps:
                        break


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
