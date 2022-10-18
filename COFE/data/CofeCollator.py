from dataclasses import dataclass
from typing import Optional, Any, Union

from transformers import PreTrainedTokenizerBase, DataCollatorForSeq2Seq
from transformers.utils import PaddingStrategy

from .data_utils import convert_to_tensors

tokenizer_accept_key = ["input_ids", 'labels', 'attention_mask']


@dataclass
class DataCollatorForCOFE:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        self.default_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            label_pad_token_id=self.label_pad_token_id,
            return_tensors=self.return_tensors
        )

    def __call__(self, features, return_tensors=None):
        # syntax of features: [list of [dataset index, feature]
        import numpy as np
        if return_tensors is None:
            return_tensors = self.return_tensors

        unpadded_features = []
        task_id = []
        # print(features, 'features in collator')
        for ds_id, feature in features:
            if ds_id == 0:
                positive_pair = feature['positive_pair']
                unpadded_features.append({k: positive_pair[k] for k in tokenizer_accept_key})
                task_id.append(0)

                negative_pair = feature['negative_pair']
                for p in negative_pair:
                    unpadded_features.append({k: p[k] for k in tokenizer_accept_key})
                    task_id.append(1)
            elif ds_id == 1:
                nli_pair = feature
                unpadded_features.append({k: nli_pair[k] for k in tokenizer_accept_key})
                task_id.append(2)

        padded_features = self.default_collator(unpadded_features,
                                                return_tensors=return_tensors)

        task_id = convert_to_tensors(task_id, return_tensors)

        padded_features['task_id'] = task_id
        return padded_features
