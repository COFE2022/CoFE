# COFE

Code for "Improving Faithfulness by Augmenting Negative Summaries from Fake Documents"

## News
This paper has been accepted by EMNLP 2022.

## COFE

### Data Preparation

#### negative data

Download prepared training negative samples for 3 datasets we mentioned from [google drive](https://drive.google.com/drive/folders/1tVxwx21GEBVfVWxU-NXWlRJekBCQyrKw?usp=sharing)

#### positive data

- xsum: in `datasets` libary
- gigaword: in `datasets` libary
- wikihow: in `datasets` libary, but need manually download the data from [here](https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag) 

#### Aligning the positive and negative data
You need to generate the positive-negative mapping file see in `COFE/data_preprocessing`
This repo provide an example for xsum (`create_pos_neg_index_xsum.sh`), you can modify it for other datasets.


### Training

see in `COFE/finetune_xsum.sh`
It use deepspeed zero3, and fp16(if your card support it) to accelerate the training.
`finetune_xsum.sh` is an example for xsum, you can modify it for other datasets.
Please note there are some comments in the script, you need to modify them according to your environment.
You need to add your own path to the ``mapping_file_path`` , ``negative_train_file`` and ``negative_train_file``

### Decoding
Please see in `COFE/decode_xsum.sh` and modify the `--resume_from_checkpoint` as your best checkpoint path.
The default output file is `COFE_predictions`, you can change it by `--save_name`'

The output follow the json format, it's a list of `[pred, gold, index]` tuple. The `index` is the index of the sample in the test-set, in order to match and for the downstream evaluation.

### Post-processing

see an example in `COFE/data_preprocessing/xsum_postprocessing.sh`
It will generate `xsum_res` dictionary and several files in it:
- res.csv store all the source, target, pred as each row
- hypo.txt & refs.txt for BERT-score
- date-dev.jsonl for factCC

### Result analysis
#### QuestEval and Rouge score
This repo provide a python script to run this part.
Please provide path to the `xsum_res` dictionary to `--result_dir` argument 
#### Coverage
Please see in this [repo](https://github.com/lil-lab/newsroom/)
here is a short example:
```python
import random

from newsroom import jsonl
from newsroom.analyze import Fragments

summary, text = "textA", "textB"
fragments = Fragments(summary, text)

# Print paper metrics:

print("Coverage:",    fragments.coverage())
print("Density:",     fragments.density())
print("Compression:", fragments.compression())

# Extractive fragments oracle:

print("List of extractive fragments:")
print(fragments.strings())
```
#### BERT-score
Please see in this [repo](https://github.com/Tiiiger/bert_score)
we used this command to get the result:
```bash
bert-score -r xsum_res/refs.txt -c xsum_res/hyps.txt --lang en --rescale_with_baseline
```

#### FactCC
We removed the factcc part because we found factcc is not very consistent with the other metircs. But you can still use it by yourself.

## Other Baselines
Please see in `baseline` folder, and follow the readme in each folder.
