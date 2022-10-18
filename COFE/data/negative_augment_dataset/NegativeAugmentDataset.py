from collections import defaultdict
from typing import Union, Dict, Optional, DefaultDict

from torch.utils.data import Dataset

from ..seq2seq_dataset import Seq2SeqDataset


class NegativeAugmentDataset(Dataset):

    def __init__(self,
                 main_dataset: Seq2SeqDataset,
                 negative_dataset: Optional[Seq2SeqDataset] = None,
                 mapping: Optional[Union[Dict, DefaultDict]] = None):
        self.main_dataset = main_dataset
        self.negative_dataset = negative_dataset
        self.mapping = defaultdict(list, mapping) if mapping is not None else defaultdict(list, {})

    def __getitem__(self, item):
        main_pair = self.main_dataset[item]
        neg_pair = [self.negative_dataset[i] for i in self.mapping[item]]
        return {
            'positive_pair': main_pair,
            'negative_pair': neg_pair
        }

    def __len__(self):
        return len(self.main_dataset)
