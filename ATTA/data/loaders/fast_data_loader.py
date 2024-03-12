# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.utils.data import DataLoader


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, sequential=False, subset=None):
        super().__init__()
        self.dataset = dataset
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        elif sequential:
            if subset is None:
                sampler = torch.utils.data.SequentialSampler(dataset)
            else:
                sampler = ActualSequentialSampler(subset)
        elif subset is not None:
            sampler = torch.utils.data.SubsetRandomSampler(subset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)
        self.sampler = sampler

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, weights, batch_size, num_workers, sequential=False, subset=None):
        super().__init__()
        self.dataset = dataset
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        elif sequential:
            if subset is None:
                sampler = torch.utils.data.SequentialSampler(dataset)
            else:
                sampler = ActualSequentialSampler(subset)
        elif subset is not None:
            sampler = torch.utils.data.SubsetRandomSampler(subset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
        self.sampler = sampler

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length


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