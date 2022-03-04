import json
import os

import datasets
import argparse
import pathlib


import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
args = parser.parse_args()

mnli_dataset = datasets.load_dataset("multi_nli")


def add_index(example, index):
    example.update({"id": index})
    return example


mnli_dataset = mnli_dataset.map(add_index, with_indices=True, num_proc=4)


def flatten_nli(example):
    return {"source": example['premise'], "target": example['hypothesis'], "label": example["label"],
            "id": example["id"]}


mnli_columns = mnli_dataset["train"].column_names
mnli_dataset = mnli_dataset.map(function=flatten_nli, remove_columns=mnli_columns, num_proc=4)


def flatten_nli2(example):
    nli_prefix = "ent" if example["label"] == 0 else "con"
    prefix = f"[{nli_prefix}] "
    return {"source": prefix + example['source'], "target": example['target'], "id": example["id"]}


mnli_columns = mnli_dataset["train"].column_names
mnli_dataset = mnli_dataset.map(function=flatten_nli2, remove_columns=mnli_columns, num_proc=1)
print(mnli_dataset)

snli = datasets.load_dataset("snli")


def add_index_snli(example, index):
    example.update({"id": index})
    return example
snli = snli.map(add_index_snli, with_indices=True,num_proc=4)




def flatten_snli(example):
    return {"source": example['premise'], "target": example['hypothesis'], "label": example["label"],"id":example["id"]}


snli_columns = snli["train"].column_names
snli = snli.map(function=flatten_snli, remove_columns=snli_columns, num_proc=4)


def flatten_snli2(example):
    nli_prefix = "ent" if example["label"] == 0 else "con"
    prefix = f"[{nli_prefix}] "
    return {"source": prefix + example['source'], "target": example['target'],"id":example["id"]}


snli_columns = snli["train"].column_names
snli = snli.map(function=flatten_snli2, remove_columns=snli_columns, num_proc=1)

print(snli)

df_mnli_train = mnli_dataset["train"].to_pandas()
df_mnli_validation = mnli_dataset["validation_matched"].to_pandas()
df_snli_train = snli["train"].to_pandas()
df_snli_validation = snli["validation"].to_pandas()

df_nli_train = pd.concat([df_mnli_train, df_snli_train], ignore_index=True, sort=False)

df_nli_validation = pd.concat([df_mnli_validation, df_snli_validation], ignore_index=True, sort=False)

print(df_nli_train)

print(pathlib.Path(args.out_dir, "train.source").as_posix())
with open(pathlib.Path(args.out_dir, "train.source").as_posix(),'w') as trainfile:
    with open(pathlib.Path(args.out_dir, "train.target").as_posix(),'w') as targetfile:
        source = df_nli_train.source.tolist()
        target = df_nli_train.target.tolist()
        for i in source:
            sent = i.replace('\n', ' ')
            trainfile.write(sent + '\n')

        for i in target:
            sent = i.replace('\n', ' ')
            targetfile.write(sent + '\n')