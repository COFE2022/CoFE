from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
from transformers import TensorType, is_tf_available, is_torch_available, is_flax_available, MODEL_MAPPING
from transformers.utils.generic import _is_jax, _is_numpy

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    'gigaword': ("document", "summary")
}


def convert_to_tensors(
        input_obj, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
):
    """
    Convert the inner content to tensors.

    Args:
        tensor_type (`str` or [`~utils.TensorType`], *optional*):
            The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
            `None`, no modification is done.
        prepend_batch_axis (`int`, *optional*, defaults to `False`):
            Whether to add the batch dimension during the conversion.
            :param prepend_batch_axis:
            :param tensor_type:
            :param input_obj:
    """
    if tensor_type is None:
        return input_obj

    # Convert to TensorType
    if not isinstance(tensor_type, TensorType):
        tensor_type = TensorType(tensor_type)

    # Get a function reference for the correct framework
    if tensor_type == TensorType.TENSORFLOW:
        if not is_tf_available():
            raise ImportError(
                "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
            )
        import tensorflow as tf

        as_tensor = tf.constant
        is_tensor = tf.is_tensor
    elif tensor_type == TensorType.PYTORCH:
        if not is_torch_available():
            raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
        import torch

        as_tensor = torch.tensor
        is_tensor = torch.is_tensor
    elif tensor_type == TensorType.JAX:
        if not is_flax_available():
            raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
        import jax.numpy as jnp  # noqa: F811

        as_tensor = jnp.array
        is_tensor = _is_jax
    else:
        as_tensor = np.asarray
        is_tensor = _is_numpy
    # (mfuntowicz: This code is unreachable)
    # else:
    #     raise ImportError(
    #         f"Unable to convert output to tensors format {tensor_type}"
    #     )

    # Do the tensor conversion in batch

    try:
        if prepend_batch_axis:
            input_obj = [input_obj]

        if not is_tensor(input_obj):
            tensor = as_tensor(input_obj)

            # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
            # # at-least2d
            # if tensor.ndim > 2:
            #     tensor = tensor.squeeze(0)
            # elif tensor.ndim < 2:
            #     tensor = tensor[None, :]

            input_obj = tensor
    except:  # noqa E722
        raise ValueError(
            "Unable to create tensor, you should probably activate truncation and/or padding "
            "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
        )

    return input_obj


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
                    "Useful for multilingual models like mBART where the first generated token"
                    "needs to be the target language token (Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def loss_label_smoothing(logits, labels, ignore_index, epsilon: float):
    log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)
    padding_mask = labels.eq(ignore_index)
    labels.clamp_min_(0)  # [b,max_len, 1]
    nll_loss = log_probs.gather(dim=-1, index=labels)
    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)
    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])

    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


mykey = "N/A" # You can get your wandb key from https://wandb.ai/authorize


def int_or_float(value):
    if '.' in str(value):
        return float(value)
    else:
        return int(value)
