# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datasets
from torch.utils.data.dataloader import default_collate

# from . import FairseqDataset
from .base_dataset import BaseDataset


class BaseWrapperDataset(BaseDataset):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def collater(cls, samples, **kwargs):
        # if hasattr(self.dataset, "collater"):
        #     return self.dataset.collater(samples, **kwargs)
        # else:
        #     return default_collate(samples)
        pass

    @property
    def sizes(self):
        return self.dataset.sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
