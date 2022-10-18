import itertools
import json
from functools import partial
from pathlib import Path
from typing import Optional, Union, List, Any
# from deepspeed.ops.adam import FusedAdam as AdamW
import datasets
import nltk
import numpy as np
import pytorch_lightning as pl

import argparse
import logging
import os

# from accelerate import Accelerator
import torch.distributed
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from datasets import load_dataset
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT, EVAL_DATALOADERS
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.text import ROUGEScore
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler, MBartTokenizer, MBartTokenizerFast, set_seed,
)

# from COFE.data.CofeCollator import DataCollatorForCOFE
from data.CofeCollator import DataCollatorForCOFE
from data.CofeDateset import CoFEDataset
from data.data_utils import add_control_code_doc, preprocess_tokenize_function, preprocessing_nli
from utils import summarization_name_mapping, MODEL_TYPES, loss_label_smoothing, mykey

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class CoFE_pl_model(pl.LightningModule):
    def __init__(self, cfg,
                 *args,
                 **kwargs
                 ):
        super(CoFE_pl_model, self).__init__()

        self.default_collator = None
        self.cofe_collator = None
        self.test_dataset = None
        self.train_dataset: CoFEDataset = None
        self.eval_dataset = None
        self.label_smoothing = None
        self.save_hyperparameters()
        self.cfg = cfg
        self.positive_weight = cfg.positive_weight
        self.negative_weight = cfg.negative_weight
        self.nli_weight = cfg.nli_weight
        if self.cfg.dataset_name is None and self.cfg.train_file is None and self.cfg.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.cfg.train_file is not None:
                extension = self.cfg.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.cfg.validation_file is not None:
                extension = self.cfg.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.cfg.val_max_target_length is None:
            self.cfg.val_max_target_length = self.cfg.max_target_length

        if any([self.cfg.nli_size, self.cfg.nli_dataset_name]) and not all(
                [self.cfg.nli_size, self.cfg.nli_dataset_name]):
            raise ValueError('nli_dataset_name and size need be set at same time')
        config_name = str(cfg.config_name).strip() if cfg.config_name else str(cfg.model_name_or_path).strip()
        self.config = AutoConfig.from_pretrained(
            config_name,
            cache_dir=cfg.cache_dir,
            # revision=cfg.model_revision,
            # use_auth_token=True if cfg.use_auth_token else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name_or_path,
            cache_dir=cfg.cache_dir,
            use_fast=cfg.use_fast_tokenizer,
            # revision=cfg.model_revision,
            # use_auth_token=True if cfg.use_auth_token else None,
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": ['[ent]', "[con]"]})
        self.model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_name_or_path,
            from_tf=bool(".ckpt" in cfg.model_name_or_path),
            config=self.config,
            cache_dir=cfg.cache_dir,
            # revision=cfg.model_revision,
            # use_auth_token=True if cfg.use_auth_token else None,
        )
        if cfg.weight_file is not None:
            self.model.from_pretrained(cfg.weight_file)

        self.model.resize_token_embeddings(len(self.tokenizer))
        if self.model.config.decoder_start_token_id is None and isinstance(tokenizer,
                                                                           (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                self.model.config.decoder_start_token_id = tokenizer.lang_code_to_id[self.lang]
            else:
                self.model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(self.self.cfg.lang)

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
                hasattr(self.model.config, "max_position_embeddings")
                and self.model.config.max_position_embeddings < cfg.max_source_length
        ):
            if cfg.resize_position_embeddings is None:
                logger.warning(
                    f"Increasing the model's number of position embedding vectors from {self.model.config.max_position_embeddings} "
                    f"to {cfg.max_source_length}."
                )
                self.model.resize_position_embeddings(cfg.max_source_length)
            elif cfg.resize_position_embeddings:
                self.model.resize_position_embeddings(cfg.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {cfg.max_source_length}, but the model only has {self.model.config.max_position_embeddings}"
                    f" position encodings. Consider either reducing `--max_source_length` to {self.model.config.max_position_embeddings} or to automatically "
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )
        self.prefix = self.cfg.source_prefix if self.cfg.source_prefix is not None else ""
        self.label_pad_token_id = -100 if self.cfg.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.weight_decay = cfg.weight_decay
        self.label_smoothing = cfg.label_smoothing

        self.ignore_index = -100 if self.cfg.ignore_pad_token_for_loss else self.tokenizer.pad_token_id

    def setup(self, stage: Optional[str] = None) -> None:
        if self.cfg.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            positive_raw_datasets = load_dataset(
                self.cfg.dataset_name,
                self.cfg.dataset_config_name,
                cache_dir=self.cfg.cache_dir,
                use_auth_token=True if self.cfg.use_auth_token else None,
            )
        else:
            positive_data_files = {}
            extension = None
            if self.cfg.train_file is not None:
                positive_data_files["train"] = self.cfg.train_file
                extension = self.cfg.train_file.split(".")[-1]
            if self.cfg.validation_file is not None:
                positive_data_files["validation"] = self.cfg.validation_file
                extension = self.cfg.validation_file.split(".")[-1]
            if self.cfg.test_file is not None:
                positive_data_files["test"] = self.cfg.test_file
                extension = self.cfg.test_file.split(".")[-1]
            positive_raw_datasets = load_dataset(
                extension,
                data_files=positive_data_files,
                cache_dir=self.cfg.cache_dir,
                use_auth_token=True if self.cfg.use_auth_token else None,
            )
        if self.cfg.do_train:
            positive_column_names = positive_raw_datasets["train"].column_names
        elif self.cfg.do_eval:
            positive_column_names = positive_raw_datasets["validation"].column_names
        elif self.cfg.do_predict:
            positive_column_names = positive_raw_datasets["test"].column_names
        elif self.cfg.do_test:
            positive_column_names = positive_raw_datasets["test"].column_names
        else:
            rank_zero_info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
            return
        sum_dataset_columns = summarization_name_mapping.get(self.cfg.dataset_name, None)
        if self.cfg.text_column is None:
            positive_text_column = sum_dataset_columns[0] if sum_dataset_columns is not None else positive_column_names[
                0]
        else:
            positive_text_column = self.cfg.text_column
            if positive_text_column not in positive_column_names:
                raise ValueError(
                    f"--text_column' value '{self.cfg.text_column}' needs to be one of: {', '.join(positive_column_names)}"
                )
        if self.cfg.summary_column is None:
            positive_summary_column = sum_dataset_columns[1] if sum_dataset_columns is not None else \
                positive_column_names[
                    1]
        else:
            positive_summary_column = self.cfg.summary_column
            if positive_summary_column not in positive_column_names:
                raise ValueError(
                    f"--summary_column' value '{self.cfg.summary_column}' needs to be one of: {', '.join(positive_column_names)}"
                )

        add_control_code_to_ori_doc = partial(add_control_code_doc,
                                              control_code='ent',
                                              text_column=sum_dataset_columns[0],
                                              summary_column=sum_dataset_columns[1])
        # with self.cfg.main_process_first(desc="add control token to positive part"):
        pos_raw_data_untokenized = positive_raw_datasets.map(function=add_control_code_to_ori_doc,
                                                             with_indices=True)
        max_source_length = self.cfg.max_source_length
        max_target_length = self.cfg.max_target_length
        padding = "max_length" if self.cfg.pad_to_max_length else False
        ignore_pad_token_for_loss = self.cfg.ignore_pad_token_for_loss
        prefix = self.cfg.source_prefix if self.cfg.source_prefix is not None else ""
        tokenizer = self.tokenizer
        preprocess_ori_function = partial(
            preprocess_tokenize_function,
            tokenizer=tokenizer,
            text_column=positive_text_column,
            summary_column=positive_summary_column,
            prefix=prefix,
            max_source_length=max_source_length,
            padding=padding,  # padding let collector do it
            max_target_length=max_target_length,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss
        )
        # with self.cfg.main_process_first(desc="train dataset map pre-processing"):
        pos_raw_data_tokenized = pos_raw_data_untokenized.map(
            preprocess_ori_function,
            with_indices=True,
            batched=True,
            num_proc=12,
            remove_columns=positive_column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )

        negative_data_files = {}
        extension = None
        if self.cfg.negative_train_file is not None:
            negative_data_files["train"] = self.cfg.negative_train_file
            extension = self.cfg.negative_train_file.split(".")[-1]
        if self.cfg.negative_validation_file is not None:
            negative_data_files["validation"] = self.cfg.negative_validation_file
            extension = self.cfg.negative_validation_file.split(".")[-1]
        if self.cfg.negative_test_file is not None:
            negative_data_files["test"] = self.cfg.negative_test_file
            extension = self.cfg.negative_test_file.split(".")[-1]
        rank_zero_info(f"{negative_data_files, extension}")
        neg_raw_data_tokenized = None
        if self.cfg.negative_train_file is not None:
            neg_raw_data_untokenized = load_dataset(
                extension,
                data_files=negative_data_files,
                cache_dir=self.cfg.cache_dir,
                use_auth_token=True if self.cfg.use_auth_token else None,
            )
            add_control_code_to_neg_doc = partial(add_control_code_doc,
                                                  control_code='con',
                                                  text_column='document',
                                                  summary_column='summary')
            # with self.cfg.main_process_first(desc="train neg dataset map add [con]"):
            neg_raw_data_untokenized = neg_raw_data_untokenized.map(function=add_control_code_to_neg_doc,
                                                                    with_indices=True)

            negative_column_names = neg_raw_data_untokenized['train'].column_names
            max_source_length = self.cfg.max_source_length
            max_target_length = self.cfg.max_target_length
            padding = "max_length" if self.cfg.pad_to_max_length else False
            ignore_pad_token_for_loss = self.cfg.ignore_pad_token_for_loss
            prefix = self.cfg.source_prefix if self.cfg.source_prefix is not None else ""
            preprocess_neg_function = partial(
                preprocess_tokenize_function,
                tokenizer=tokenizer,
                text_column='document',
                summary_column='summary',
                prefix=prefix,
                max_source_length=max_source_length,
                padding=padding,  # padding let collector do it
                max_target_length=max_target_length,
                ignore_pad_token_for_loss=ignore_pad_token_for_loss
            )
            # with self.cfg.main_process_first(desc="train negative dataset map pre-processing"):
            neg_raw_data_tokenized = neg_raw_data_untokenized.map(
                preprocess_neg_function,
                with_indices=True,
                batched=True,
                num_proc=12,
                remove_columns=positive_column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on negative dataset",
            )
        nli_train = None
        if len(self.cfg.nli_dataset_name) > 0:
            dsets = []
            for nli_name in self.cfg.nli_dataset_name:
                ds = load_dataset(nli_name)
                # with self.cfg.main_process_first(desc="nli dataset map pre-processing"):
                ds = preprocessing_nli(ds)
                dsets.append(ds['train'])
            nli_raw_dataset = datasets.concatenate_datasets(dsets)
            max_source_length = self.cfg.max_source_length
            max_target_length = self.cfg.max_target_length
            padding = "max_length" if self.cfg.pad_to_max_length else False
            ignore_pad_token_for_loss = self.cfg.ignore_pad_token_for_loss
            prefix = self.cfg.source_prefix if self.cfg.source_prefix is not None else ""
            # with self.cfg.main_process_first(desc="train dataset map pre-processing"):
            preprocess_nli_function = partial(
                preprocess_ori_function,
                tokenizer=tokenizer,
                text_column='source',
                summary_column='target',
                prefix=prefix,
                max_source_length=max_source_length,
                padding=padding,  # padding let collector do it
                max_target_length=max_target_length,
                ignore_pad_token_for_loss=ignore_pad_token_for_loss
            )
            column_names = nli_raw_dataset.column_names
            nli_train = nli_raw_dataset.map(
                preprocess_nli_function,
                with_indices=True,
                batched=True,
                num_proc=12,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on nli dataset",
            )
        # if self.cfg.max_train_samples is not None:
        #     max_train_samples = min(len(pos_raw_data_tokenized['train']), self.cfg.max_train_samples)
        #     pos_raw_data_tokenized['train'] = pos_raw_data_tokenized['train'].select(range(max_train_samples))
        # if self.cfg.max_eval_samples is not None:
        #     max_eval_samples = min(len(pos_raw_data_tokenized['validation']), self.cfg.max_eval_samples)
        #     pos_raw_data_tokenized['validation'] = pos_raw_data_tokenized.select(range(max_eval_samples))
        # if self.cfg.max_predict_samples is not None:
        #     max_predict_samples = min(len(pos_raw_data_tokenized['test']), self.cfg.max_predict_samples)
        #     pos_raw_data_tokenized['test'] = pos_raw_data_tokenized['test'].select(range(max_predict_samples))

        self.train_dataset = CoFEDataset(
            positive_dataset=pos_raw_data_tokenized['train'],
            negative_dataset=neg_raw_data_tokenized['train'] if self.cfg.negative_train_file is not None else None,
            pos_neg_mapping_path=self.cfg.mapping_file_path if self.cfg.mapping_file_path is not None else None,
            nli_dataset=nli_train if self.cfg.nli_dataset_name is not None else None,
            nli_size=self.cfg.nli_size if self.cfg.nli_size is not None else None,
            nli_sampling_ratios=1,
        )
        self.eval_dataset = pos_raw_data_tokenized['validation']
        self.test_dataset = pos_raw_data_tokenized['test']
        #  stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        #
        #     self.eval_dataset = pos_raw_data_tokenized['validation']
        # elif stage == 'test' or stage == 'predict':
        #     self.test_dataset = pos_raw_data_tokenized['test']

        if self.cfg.label_smoothing > 0 and not hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{self.model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )
            self.train_dataset = CoFEDataset(
                positive_dataset=pos_raw_data_tokenized['train'],
                negative_dataset=neg_raw_data_tokenized['train'] if self.cfg.negative_train_file is not None else None,
                pos_neg_mapping_path=self.cfg.mapping_file_path if self.cfg.mapping_file_path is not None else None,
                nli_dataset=nli_train if self.cfg.nli_dataset_name is not None else None,
                nli_size=self.cfg.nli_size if self.cfg.nli_size is not None else None,
                nli_sampling_ratios=1,
            )

            # Data collator
        label_pad_token_id = -100 if self.cfg.ignore_pad_token_for_loss else tokenizer.pad_token_id
        self.cofe_collator = DataCollatorForCOFE(
            tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.cfg.fp16 else None,
        )
        self.default_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.cfg.fp16 else None,
        )
        self.configure_metrics(stage)
        if self.train_dataset is not None:
            logger.info(f"the training dataset size is {len(self.train_dataset)}")
        self.res = []

        rank_zero_info(f'size of test dataset is {len(self.test_dataset)}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.per_device_train_batch_size,
            collate_fn=self.cofe_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.per_device_eval_batch_size,
            collate_fn=self.default_collator,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.per_device_eval_batch_size,
            collate_fn=self.default_collator,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.per_device_eval_batch_size,
            collate_fn=self.default_collator,
        )

    def forward(self, batch, batch_idx):

        return self.model(**batch)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          self.cfg.learning_rate)
        # todo add beta
        end_steps = self.cfg.max_steps if self.cfg.max_steps else self.trainer.max_steps
        estimated_max_steps = (len(self.train_dataset) * self.trainer.max_epochs) // (
                self.trainer.world_size * self.trainer.accumulate_grad_batches * self.cfg.per_device_train_batch_size) \
            if self.train_dataset is not None else 0
        num_training_steps = self.cfg.max_steps if self.cfg.max_steps != -1 else estimated_max_steps
        rank_zero_info(f" the num_training_steps is {num_training_steps} ")
        lr_scheduler = get_scheduler(
            name=self.cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.num_warmup_steps,
            num_training_steps=num_training_steps
        )
        lr_dict = {
            # REQUIRED: The scheduler instance
            'scheduler': lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            'interval': 'step',
            # # How many epochs/steps should pass between calls to
            # # `scheduler.step()`. 1 corresponds to updating the learning
            # # rate after every epoch/step.
            # 'frequency': 1,
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_dict
        }

    def _nll_loss(self, some_logits, labels):
        if self.label_smoothing is None or self.label_smoothing == 0.0:
            log_probs = -nn.functional.log_softmax(some_logits, dim=-1)
            if labels.dim() == log_probs.dim() - 1:
                labels = labels.unsqueeze(-1)
            padding_mask = labels.eq(self.ignore_index)
            labels = torch.clamp(labels, min=0)
            nll_loss = log_probs.gather(dim=-1, index=labels)
            # works for fp16 input tensor too, by internally upcasting it to fp32
            # smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
            nll_loss.masked_fill_(padding_mask, 0.0)
            num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            nll_loss = nll_loss.sum() / num_active_elements
            return nll_loss
        else:
            return loss_label_smoothing(some_logits, labels, self.ignore_index, self.label_smoothing)

    def _loss(self, inputs):
        index = inputs.pop('index', None)
        task_mask = inputs.pop('task_id', None)
        model_outputs = self.model(**inputs)
        self.log('original_train_loss', model_outputs.loss.item(), on_epoch=True, on_step=True)
        logits = model_outputs.logits
        labels = inputs['labels']
        original_ids = task_mask == 0
        original_logits = logits[original_ids]
        original_labels = labels[original_ids]
        negative_ids = task_mask == 1
        negative_logits = logits[negative_ids]
        negative_labels = labels[negative_ids]
        nli_ids = task_mask == 2
        nli_logits = logits[nli_ids]
        nli_labels = labels[nli_ids]
        loss = 0
        if len(original_labels) > 0:
            positive_loss = self._nll_loss(original_logits, original_labels)
            positive_loss = positive_loss * self.positive_weight
            loss += positive_loss

        if len(negative_logits):
            negative_loss = self._nll_loss(negative_logits, negative_labels)
            negative_loss = negative_loss * self.negative_weight
            loss += negative_loss
        if len(nli_labels):
            nli_loss = self._nll_loss(nli_logits, nli_labels)
            nli_loss = self.nli_weight * nli_loss
            loss += nli_loss
        return loss

    def training_step(self, batch, batch_idx):
        # index = batch.pop('index', None)
        # task_id = batch.pop('task_id', None)
        # model_output = self.model(**batch)
        # rank_zero_info(self.tokenizer.batch_decode(batch['input_ids'], ))

        loss = self._loss(batch)
        self.log("loss", loss, on_step=True, on_epoch=True)
        return loss
        # if self.label_smoothing == 0:
        #     loss = result.loss
        # else:
        #     labels = batch["labels"]
        #     loss = loss_label_smoothing(result, labels, self.label_pad_token_id, self.label_smoothing)
        #     self.log('original_train_loss', result.loss, on_epoch=True, on_step=True)
        # self.log('train_loss', loss, on_epoch=True, on_step=True)
        # return loss

    def configure_metrics(self, stage: str):
        self.rouge = ROUGEScore(use_stemmer=True)

    def _generate(self, input_ids, attention_mask, labels):
        gen_kwargs = {
            "max_length": self.cfg.val_max_target_length,
            "num_beams": self.cfg.num_beams,
        }
        generated_tokens = self.model.generate(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               **gen_kwargs)
        # labels = batch["labels"]
        # ids = batch['index']

        # def pad_across_processes(data: torch.Tensor, dim=0, pad_index=0, pad_first=False):
        #     size = torch.tensor(data.shape, device=self.device)
        #     # size = generated_tokens.shape
        #     sizes = self.all_gather(size).cpu()
        #     if sizes.dim() > 1:
        #         max_size = max(s[dim].item() for s in sizes)
        #     else:
        #         max_size = sizes[dim].item()
        #     if max_size == data.shape[dim]:
        #         return data
        #     old_size = data.shape
        #     new_size = list(old_size)
        #     new_size[dim] = max_size
        #     new_tensor = data.new_zeros(tuple(new_size)) + pad_index
        #     if pad_first:
        #         indices = tuple(
        #             slice(max_size - old_size[dim], max_size) if i == dim else slice(None) for i in range(len(new_size))
        #         )
        #     else:
        #         indices = tuple(slice(0, old_size[dim]) if i == dim else slice(None) for i in range(len(new_size)))
        #     new_tensor[indices] = data
        #     return new_tensor

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        if self.cfg.ignore_pad_token_for_loss:
            labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels,

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        return decoded_preds, decoded_labels

    def validation_step(self, batch, batch_idx):
        decoded_preds, decoded_labels = self._generate(batch['input_ids'], batch['attention_mask'], batch['labels'])
        # index = batch.pop('index', None)
        # task_id = batch.pop('task_id', None)
        ids = batch.pop('index', None)
        ids = ids.cpu().numpy().tolist()
        output = self.model(**batch)
        loss = output.loss
        self._log_rouge(decoded_preds, decoded_labels, 'val')
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        out = [{'decoded_preds': decoded_preds[i],
                'decoded_label': decoded_labels[i],
                'id': ids[i]} for i in
               range(len(ids))]
        return out

    def _log_rouge(self, decoded_preds, decoded_labels, prefix=''):
        result = self.rouge(decoded_preds, decoded_labels)
        res2 = {}
        res2[f'{prefix}_rouge1'] = result['rouge1_fmeasure'] * 100
        res2[f'{prefix}_rouge2'] = result['rouge2_fmeasure'] * 100
        res2[f'{prefix}_rougeL'] = result['rougeL_fmeasure'] * 100
        res2[f'{prefix}_rougeLsum'] = result['rougeLsum_fmeasure'] * 100
        self.log(f'{prefix}_rouge1', res2[f'{prefix}_rouge1'], on_step=False, on_epoch=True)
        self.log(f'{prefix}_rouge2', res2[f'{prefix}_rouge2'], on_step=False, on_epoch=True)
        self.log(f'{prefix}_rougeL', res2[f'{prefix}_rougeL'], on_step=False, on_epoch=True)
        self.log(f'{prefix}_rougeLsum', res2[f'{prefix}_rougeLsum'], on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = [j for i in outputs for j in i]
        if self.trainer.world_size > 1:
            temp = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(temp, outputs)
            outputs = temp
            outputs = [j for i in outputs for j in i]
        if self.global_rank == 0:
            with open(f"val_{self.cfg.negative_weight}_{self.trainer.current_epoch}_{self.cfg.save_name}",
                      "w") as writer:
                json.dump(outputs, writer)
        rank_zero_info('saved!')

    def test_step(self, batch, batch_idx):
        decoded_preds, decoded_labels = self._generate(batch['input_ids'], batch['attention_mask'], batch['labels'])
        # index = batch.pop('index', None)
        task_id = batch.pop('task_id', None)
        ids = batch.pop('index', None)
        ids = ids.cpu().numpy().tolist()
        output = self.model(**batch)
        loss = output.loss
        self._log_rouge(decoded_preds, decoded_labels, 'val')
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        out = [{'decoded_preds': decoded_preds[i],
                'decoded_label': decoded_labels[i],
                'id': ids[i]} for i in
               range(len(ids))]
        return out

    @rank_zero_only
    def _store_output(self, ):
        with open(f"{self.cfg.save_name}", "w") as outputfile:
            json.dump(self.res, outputfile)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = [j for i in outputs for j in i]
        if self.trainer.world_size > 1:
            temp = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(temp, outputs)
            outputs = temp
            outputs = [j for i in outputs for j in i]
        if self.global_rank == 0:
            with open(f"{self.cfg.save_name}", "w") as writer:
                json.dump(outputs, writer)
        rank_zero_info('saved!')

    def on_predict_end(self) -> None:
        self._store_output()

    def on_train_epoch_start(self) -> None:
        self.train_dataset.set_epoch(self.trainer.current_epoch)
        rank_zero_info(f'{self.trainer.train_dataloader.dataset}')

    def _create_res_list(self):
        self.res = []

    def on_predict_start(self) -> None:
        self._create_res_list()

    def on_test_start(self) -> None:
        self._create_res_list()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        decoded_preds, decoded_labels = self._generate_with_store(batch, batch_idx, dataloader_idx)
        result = self.rouge(decoded_preds, decoded_labels)
        res2 = {}
        res2['rouge1'] = result['rouge1_fmeasure'] * 100
        res2['rouge2'] = result['rouge2_fmeasure'] * 100
        res2['rougeL'] = result['rougeL_fmeasure'] * 100
        res2['rougeLsum'] = result['rougeLsum_fmeasure'] * 100
        self.log('predict_rouge1', res2['rouge1'], on_step=False, on_epoch=True)
        self.log('predict_rouge2', res2['rouge2'], on_step=False, on_epoch=True)
        self.log('predict_rougeL', res2['rougeL'], on_step=False, on_epoch=True)
        self.log('predict_rougeLsum', res2['rougeLsum'], on_step=False, on_epoch=True)

    def _generate_with_store(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        decoded_preds, decoded_labels, ids = self._generate(batch)
        self.res = self.res + list(zip(decoded_preds, decoded_labels, ids.cpu().numpy().tolist()))
        return decoded_preds, decoded_labels

    @staticmethod
    def add_model_specific_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--label_smoothing",
            type=float,
            default=0,
            help="if 0 then use default loss from model.forward, otherwise compute loss twice",
        )
        parser.add_argument(
            "--clip_norm",
            type=float,
            default=0.0,
            help="clip gradient respect to norm of weight in case of gradient explosion",
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="The name of the dataset to use (via the datasets library).",
        )
        parser.add_argument(
            "--dataset_config_name",
            type=str,
            default=None,
            help="The configuration name of the dataset to use (via the datasets library).",
        )
        parser.add_argument(
            "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
        )
        parser.add_argument(
            "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
        )
        parser.add_argument(
            "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
        )
        parser.add_argument(
            "--ignore_pad_token_for_loss",
            type=bool,
            default=True,
            help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
        )
        parser.add_argument(
            "--max_source_length",
            type=int,
            default=1024,
            help="The maximum total input sequence length after "
                 "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--source_prefix",
            type=str,
            default=None,
            help="A prefix to add before every source text " "(useful for T5 models).",
        )
        parser.add_argument(
            "--preprocessing_num_workers",
            type=int,
            default=None,
            help="The number of processes to use for the preprocessing.",
        )
        parser.add_argument(
            "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument(
            "--max_target_length",
            type=int,
            default=128,
            help="The maximum total sequence length for target text after "
                 "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
                 "during ``evaluate`` and ``predict``.",
        )
        parser.add_argument(
            "--val_max_target_length",
            type=int,
            default=None,
            help="The maximum total sequence length for validation "
                 "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
                 "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
                 "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
        )
        parser.add_argument(
            "--max_length",
            type=int,
            default=128,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
            ),
        )
        parser.add_argument(
            "--num_beams",
            type=int,
            default=None,
            help="Number of beams to use for evaluation. This argument will be "
                 "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
        )
        parser.add_argument(
            "--pad_to_max_length",
            action="store_true",
            help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
        )
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
            required=True,
        )
        parser.add_argument(
            "--config_name",
            type=str,
            default=None,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            type=str,
            default=None,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--text_column",
            type=str,
            default=None,
            help="The name of the column in the datasets containing the full texts (for summarization).",
        )
        parser.add_argument(
            "--summary_column",
            type=str,
            default=None,
            help="The name of the column in the datasets containing the summaries (for summarization).",
        )

        parser.add_argument(
            "--use_slow_tokenizer",
            action="store_true",
            help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
        )
        parser.add_argument(
            "--per_device_train_batch_size",
            type=int,
            default=8,
            help="Batch size (per device) for the training dataloader.",
        )
        parser.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=8,
            help="Batch size (per device) for the evaluation dataloader.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=3e-5,
            help="Initial learning rate (after the potential warmup period) to use.",
        )
        parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
        parser.add_argument(
            "--lr_scheduler_type",
            type=str,
            default="linear",
            help="The scheduler type to use.",
            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        )
        parser.add_argument(
            "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
        )
        parser.add_argument(
            "--model_type",
            type=str,
            default=None,
            help="Model type to use if training from scratch.",
            choices=MODEL_TYPES,
        )
        parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
        parser.add_argument(
            "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
        )
        parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
        parser.add_argument(
            "--checkpointing_steps",
            type=str,
            default=None,
            help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
        )
        parser.add_argument(
            "--with_tracking",
            action="store_true",
            help="Whether to load in all available experiment trackers from the environment and use them for logging.",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Where to store the pretrained models downloaded from huggingface.co",
        )
        parser.add_argument(
            "--model_revision",
            type=str,
            default="main",
            help="The specific model version to use (can be a branch name, tag name or commit id).",
        )
        parser.add_argument(
            "--use_auth_token",
            action='store_false',
            help="Will use the token generated when running `transformers-cli login` (necessary to use this script "
                 "with private models).",
        )
        parser.add_argument(
            "--use_fast_tokenizer",
            action='store_true',
            help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
        )

        parser.add_argument(
            '--positive_weight', default=1.0, type=float, help="The lambda weight coefficients of main task."
        )
        parser.add_argument(
            '--negative_weight', default=1.0, type=float, help="The lambda weight coefficients of negative part."
        )
        parser.add_argument(
            '--nli_weight', default=1.0, type=float, help="The lambda weight coefficients of negative part."
        )
        parser.add_argument(
            '--negative_train_file', default=None, type=str,
            help="The input negative training data file (a jsonlines or csv file)."
        )
        parser.add_argument(
            '--negative_validation_file', default=None, type=str,
            help="The input negative val data file (a jsonlines or csv file)."
        )
        parser.add_argument(
            '--negative_test_file', default=None, type=str,
            help="The input negative test data file (a jsonlines or csv file)."
        )
        parser.add_argument(
            '--mapping_file_path', default=None, type=str,
            help="The pickle file path of mapping positive sample and negative sample"
        )

        parser.add_argument(
            '--nli_dataset_name', default=[], nargs='+',
            help="the list of datasets name for nli."
        )

        parser.add_argument(
            '--nli_size', default=None, type=int,
            help="resize the nli data size to the specified value."
        )

        parser.add_argument(
            '--save_name', type=str, default='COFE_predictions.txt',
        )

        return parser
