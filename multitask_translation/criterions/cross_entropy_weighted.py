# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import torch.nn.functional as F
from omegaconf import II

logger = logging.getLogger(__name__)


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def loss_label_smoothing_weighted(log_prob, label, ignore_index=None, label_smoothing: float = 0, weights=None):
    if weights is None:
        weights = torch.ones(label.shape[0], device=label.device).view(-1, 1)
    else:
        weights = weights.view(-1, 1)
    label = label.clone().detach()
    if label.dim() == log_prob.dim() - 1:
        label = label.unsqueeze(-1)

    if ignore_index is not None:
        padding_mask = label.eq(ignore_index)
        label.clamp_min_(0)  # [b,max_len, 1]
        nll_loss = -log_prob.gather(dim=-1, index=label)
        # works for fp16 input tensor too, by internally upcasting it to fp32

        smoothed_loss = log_prob.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        nll_loss = nll_loss.squeeze(-1)
        smoothed_loss = smoothed_loss.squeeze(-1)
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        active_elements = torch.ones_like(nll_loss) - padding_mask.squeeze(-1).long()
        active_elements = active_elements * weights
        weighted_nll_loss = nll_loss * weights
        weighted_nll_loss = weighted_nll_loss.sum() / active_elements.sum()
        smoothed_loss = smoothed_loss.sum() / (active_elements.sum() * log_prob.shape[-1])
        total_loss = (1 - label_smoothing) * weighted_nll_loss + label_smoothing * smoothed_loss
        nll_loss = nll_loss.sum()/(torch.ones_like(nll_loss) - padding_mask.squeeze(-1).long()).sum()
        return total_loss, nll_loss
    else:
        nll_loss = log_prob.gather(dim=-1, index=label)
        smoothed_loss = log_prob.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss = nll_loss.squeeze(-1)
        smoothed_loss = smoothed_loss.squeeze(-1)
        active_elements = torch.ones_like(nll_loss)
        weighted_nll_loss = nll_loss * weights
        weighted_nll_loss = weighted_nll_loss.sum() / active_elements.sum()
        smoothed_loss = smoothed_loss.sum() / (active_elements.sum() * log_prob.shape[-1])
        total_loss = weighted_nll_loss * (1.0 - label_smoothing) + label_smoothing * smoothed_loss
        nll_loss = nll_loss.mean()
        return total_loss, nll_loss


@register_criterion(
    "cross_entropy_weighted", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyWeightedCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        # logger.info(net_output.shape)
        loss, nll_loss = self.compute_loss(model, net_output, sample)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        max_ds_id = sample['dataset_id'].max()
        # logger.info(max_ds_id)
        datasets_count = {i: (sample['dataset_id'] == i).sum().item() for i in range(max_ds_id.item()+1)}

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        # logger.info(datasets_count)
        for k, v in datasets_count.items():
            logging_output["ds" + f"{k}"] = v

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_lprobs_and_target_no_flatten(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss_weighted(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target_no_flatten(model, net_output, sample)
        # lprobes = [b,maxlen,vocab] target[b,max_tgt_len]

        loss, nll_loss = loss_label_smoothing_weighted(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            weights=sample['weights'] if hasattr(sample, 'weights') else None,
            label_smoothing=self.eps
        )
        return loss, nll_loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        # logger.info(net_output[0].shape)
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # logger.info(lprobs.shape)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
