import argparse
import json
import logging
import pickle
from collections import OrderedDict
from functools import partial
from pprint import pprint
from typing import Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from .CofeCollator import DataCollatorForCOFE
from .data_utils import preprocessing_nli, add_control_code_doc, preprocess_tokenize_function, summarization_name_mapping
from .multitask_dataset import ResamplingDataset, SampledMultiDataset
from .negative_augment_dataset import NegativeAugmentDataset
from .seq2seq_dataset import Seq2SeqDataset

import datasets

logger = logging.getLogger(__name__)
datasets.logging.set_verbosity_error()


class CoFEDataset(Dataset):

    def __init__(self,
                 positive_dataset: datasets.Dataset,
                 negative_dataset: Optional[datasets.Dataset] = None,
                 pos_neg_mapping_path: Optional[str] = None,
                 nli_dataset: Optional[datasets.Dataset] = None,
                 nli_sampling_ratios: Optional[float] = 1.0,
                 nli_size: Optional[int] = None,
                 ):
        self.dataset = None
        positive_dataset: Seq2SeqDataset = Seq2SeqDataset(positive_dataset)
        self.nli_dataset = nli_dataset
        self.pos_neg_mapping = pos_neg_mapping_path
        self.sampling_ratios = nli_sampling_ratios
        self.nli_size = nli_size
        datasets_dict = None
        sampling_rate_dict = None
        if (negative_dataset is not None and pos_neg_mapping_path is None) or (
                negative_dataset is None and pos_neg_mapping_path is not None):
            raise ValueError('one of negative_dataset or mapping is None')

        if negative_dataset is not None:
            negative_dataset = Seq2SeqDataset(negative_dataset)
            with open(pos_neg_mapping_path, 'rb') as mapping_file:
                pair_mapping = pickle.load(mapping_file)
            self.pair_mapping = pair_mapping
            self.negative_dataset = Seq2SeqDataset(negative_dataset)
            paired_sum_dataset = NegativeAugmentDataset(positive_dataset,
                                                        negative_dataset,
                                                        mapping=self.pair_mapping)
            datasets_dict = OrderedDict({"paired_sum_dataset": paired_sum_dataset})
            sampling_rate_dict = OrderedDict({"paired_sum_dataset": 1})
            if nli_dataset is not None:
                self.nli_dataset = Seq2SeqDataset(nli_dataset)
                assert nli_sampling_ratios is not None
                assert nli_size is not None

                # in most cases, the size of the nli dataset is far bigger than the size of the main dataset
                nli_ratio: float = nli_size / len(nli_dataset)
                nli_ratio = min(1, nli_ratio)
                if nli_ratio < 1.0:
                    logger.info(f"size of nli is {len(self.nli_dataset)}, resample it to {nli_size}")
                    self.nli_dataset = ResamplingDataset(self.nli_dataset, replace=False, size_ratio=nli_ratio)
                elif nli_ratio == 1:
                    logger.info(f"size of nli is {len(self.nli_dataset)}, degrade to seq2seq_dataset")
                datasets_dict.update({'nli': self.nli_dataset})
                sampling_rate_dict.update({'nli': nli_sampling_ratios})
        elif negative_dataset is None:
            # for the negative dataset is null case, then degrade to the empty case
            datasets_dict = OrderedDict({"paired_sum_dataset": NegativeAugmentDataset(positive_dataset)})
            sampling_rate_dict = OrderedDict({"paired_sum_dataset": 1})
        self.dataset = SampledMultiDataset(datasets_dict, sampling_rate_dict)

    def set_epoch(self,epoch_id):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--neg_file_path', type=str, help='file path to the generated negative csv', required=True)
#     parser.add_argument('--source_column', type=str, help='source column in header')
#     parser.add_argument('--target_column', type=str, help='target column in header')
#     parser.add_argument('--dataset_name', type=str, help='dataset name for loading "load_dataset" ', required=True)
#     parser.add_argument('--dataset_config_name', type=str, help='dataset name for loading "load_dataset"', default=None)
#     args = parser.parse_args()
#     tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('facebook/bart-large')
#     tokenizer.add_special_tokens({"additional_special_tokens": ['[ent]', "[con]"]})
#
#     # positive part
#     dataset_name = 'xsum_data'
#     pos_raw_data_untokenized = datasets.load_dataset(dataset_name)
#     dataset_columns = summarization_name_mapping.get(dataset_name, None)
#     add_control_code_to_ori_doc = partial(add_control_code_doc,
#                                           control_code='ent',
#                                           text_column=dataset_columns[0],
#                                           summary_column=dataset_columns[1])
#     pos_raw_data_untokenized = pos_raw_data_untokenized.map(function=add_control_code_to_ori_doc, with_indices=True)
#
#     # negative part
#     data_files = {}
#     if args.neg_file_path is not None:
#         data_files["train"] = args.neg_file_path
#     extension = args.neg_file_path.split(".")[-1]
#     neg_raw_data_untokenized = datasets.load_dataset(extension, data_files=data_files)
#     add_control_code_to_neg_doc = partial(add_control_code_doc,
#                                           control_code='con',
#                                           text_column='document',
#                                           summary_column='summary')
#     neg_raw_data_untokenized = neg_raw_data_untokenized.map(function=add_control_code_to_neg_doc, with_indices=True)
#
#     # nli part
#     nli_raw_dataset = datasets.load_dataset('multi_nli')
#     nli_raw_dataset = preprocessing_nli(nli_raw_dataset)
#
#     mapping_file_path = '../../data_preprocessing/mapping_files/xsum_mapping_file.pickle'
#     with open(mapping_file_path, 'rb') as mapping_file:
#         pair_mapping = pickle.load(mapping_file)
#
#     # start of collecting sub dataset
#     nli_size = 50000
#     sum_dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
#
#     # start of tokenizing function
#     column_names = sum_dataset_columns
#
#     preprocess_ori_function = partial(
#         preprocess_tokenize_function,
#         tokenizer=tokenizer,
#         text_column=sum_dataset_columns[0],
#         summary_column=sum_dataset_columns[1],
#         prefix='',
#         max_source_length=1024,
#         padding=False,  # padding let collector do it
#         max_target_length=128,
#         ignore_pad_token_for_loss=True
#     )
#
#     pos_raw_data_tokenized = pos_raw_data_untokenized.map(
#         preprocess_ori_function,
#         with_indices=True,
#         batched=True,
#         num_proc=12,
#         remove_columns=column_names,
#         load_from_cache_file=True,
#         desc="Running tokenizer on train dataset",
#     )
#     preprocess_neg_function = partial(
#         preprocess_ori_function,
#         tokenizer=tokenizer,
#         text_column='document',
#         summary_column='summary',
#         prefix='',
#         max_source_length=1024,
#         padding=False,  # padding let collector do it
#         max_target_length=128,
#         ignore_pad_token_for_loss=True
#     )
#
#     column_names = neg_raw_data_untokenized['train'].column_names
#     neg_raw_data_tokenized = neg_raw_data_untokenized.map(
#         preprocess_neg_function,
#         with_indices=True,
#         batched=True,
#         num_proc=12,
#         remove_columns=column_names,
#         load_from_cache_file=True,
#         desc="Running tokenizer on negative dataset",
#     )
#
#     preprocess_nli_function = partial(
#         preprocess_ori_function,
#         tokenizer=tokenizer,
#         text_column='source',
#         summary_column='target',
#         prefix='',
#         max_source_length=1024,
#         padding=False,  # padding let collector do it
#         max_target_length=128,
#         ignore_pad_token_for_loss=True
#     )
#     column_names = nli_raw_dataset['train'].column_names
#     nli_raw_dataset = nli_raw_dataset.map(
#         preprocess_nli_function,
#         with_indices=True,
#         batched=True,
#         num_proc=12,
#         remove_columns=column_names,
#         load_from_cache_file=True,
#         desc="Running tokenizer on nli dataset",
#     )
#
#     pos_dataset = Seq2SeqDataset(pos_raw_data_tokenized['train'])
#     neg_dataset = Seq2SeqDataset(neg_raw_data_tokenized['train'])
#     paired_sum_dataset = NegativeAugmentDataset(pos_dataset, neg_dataset, pair_mapping)
#     nli_dataset = ResamplingDataset(nli_raw_dataset['train'], size_ratio=2)
#
#     datasets_dict = OrderedDict({"paired_sum_dataset": paired_sum_dataset,
#                                  "nli": nli_dataset})
#     sampling_rate_dict = OrderedDict({"paired_sum_dataset": 1,
#                                       "nli": 1})
#     mix_dataset = SampledMultiDataset(datasets_dict,
#                                       sampling_ratios=sampling_rate_dict,
#                                       )
#     #
#     # pprint(tokenizer.decode(pos_dataset[0]['input_ids']))
#     # pprint(tokenizer.decode(neg_dataset[0]['input_ids']))
#     # pprint(tokenizer.decode(nli_dataset[3]['input_ids']))
#
#     data_collator = DataCollatorForCOFE(
#         tokenizer,
#         model=None,
#         padding=True,
#         label_pad_token_id=-100,
#         pad_to_multiple_of=8,
#     )
#     res = data_collator([mix_dataset[0]])
#     # pprint(tokenizer.batch_decode(data_collator([mix_dataset[i] for i in range(8)])['labels']))
#     pos_data_untokenized = Seq2SeqDataset(pos_raw_data_untokenized['train'])
#     neg_data_untokenized = Seq2SeqDataset(neg_raw_data_untokenized['train'])
#     paired_sum_dataset_untokenized = NegativeAugmentDataset(pos_data_untokenized, neg_data_untokenized, pair_mapping)
#     # pprint(paired_sum_dataset_untokenized[0])
#     # pprint(pos_data_untokenized[0])
#     # pprint(neg_data_untokenized[0])
#     # pprint('################################')
#     # pprint(paired_sum_dataset_untokenized[0])
#     # print('################################')
#     # pprint(pos_raw_data_untokenized['train'][1145])
#     # pprint(tokenizer.decode(pos_raw_data_tokenized['train'][1145]['input_ids']))
#     # pprint(neg_raw_data_untokenized['train'][1145])
#     # pprint(tokenizer.decode(neg_raw_data_tokenized['train'][1145]['input_ids']))
#     # pprint(paired_sum_dataset_untokenized[0])
#     # pprint(tokenizer.decode(paired_sum_dataset[0]['positive_pair']['labels']))
#     # pprint(tokenizer.decode(paired_sum_dataset[0]['negative_pair'][0]['labels']))
#     pprint(tokenizer.batch_decode(res['input_ids']))
#     mix_raw_data_untoken = SampledMultiDataset(
#         OrderedDict({"paired_sum_dataset": paired_sum_dataset_untokenized,
#                      "nli": nli_dataset}),
#         sampling_ratios=sampling_rate_dict
#     )
#     pprint(mix_raw_data_untoken[0])
#     c = CoFEDataset(
#         positive_dataset=pos_raw_data_tokenized['train'],
#         negative_dataset=neg_raw_data_tokenized['train'],
#         pos_neg_mapping_path=mapping_file_path,
#         nli_dataset=nli_raw_dataset['train'],
#         nli_size=nli_size,
#         nli_sampling_ratios=1,
#     )
#
#     r = data_collator([c[200000], c[1], c[10]])
