# -*- coding: utf-8 -*-
import argparse
import os

import nltk
import numpy as np
import pandas as pd
from datasets import load_metric, load_dataset
import json
import pandas
from nltk import ngrams

parser = argparse.ArgumentParser()
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
    "--cache_dir",
    type=str,
    default=None,
    help="Where to store the pretrained models downloaded from huggingface.co",
)
parser.add_argument('--jsonfile_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--split', type=str, default='test')
args = parser.parse_args()
rouge_metric = load_metric("rouge")
resfile = args.jsonfile_path
outputdir_name = args.output_dir if args.output_dir else f"{args.dataset_name}_out"
split = args.split

os.makedirs(f"{os.getcwd()}/{outputdir_name}", exist_ok=True)
save_path = f"{os.getcwd()}/{outputdir_name}/data-dev.jsonl"
save_path2 = f"{os.getcwd()}/{outputdir_name}/res.csv"
refs_path = f"{os.getcwd()}/{outputdir_name}/refs.txt"
hypes_path = f"{os.getcwd()}/{outputdir_name}/hypes.txt"
# print(save_path)
with open(resfile, "r") as infile:
    res = json.load(infile)

labels = [i['decoded_label'] for i in res]
preds = [i['decoded_preds'] for i in res]
ids = [i['id'] for i in res]
print(res[0])
ds = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )


def add_index(example, index):
    example.update({"index": index})
    return example


ds = ds.map(add_index, with_indices=True)
ds_split = ds[split]
df = pd.DataFrame(data={"preds": preds, "labels": labels, "index": ids})
srcs = []
tgts = []
ids = []

for example in ds_split:
    srcs.append(example["document"])
    tgts.append(example["summary"])
    ids.append(int(example["index"]))
df2 = pd.DataFrame(data={"srcs": srcs, "tgts": tgts, "index": ids})

result = pd.concat([df, df2], join="inner")
r2 = pd.merge(df, df2, how='inner', on="index")
r2 = r2.sort_values("index")
r2 = r2.drop_duplicates(subset=['index'])
r2.reindex()
with open(save_path, "w") as file:
    for e in r2.iterrows():
        # print(e[-1])
        e = e[-1]
        line = {}
        line["id"] = e["index"]
        line["text"] = e["srcs"]
        line["claim"] = e["preds"]
        line["label"] = "CORRECT"
        file.write(json.dumps(line) + "\n")

r2.to_csv(save_path2, index=False, encoding="utf-8")

with open(refs_path, "w", encoding="utf-8") as refsout:
    with open(hypes_path, "w", encoding="utf-8") as hypsout:
        for i in r2.iterrows():
            i = i[-1]
            ref: str = i["srcs"]
            hyp = i["preds"]
            ref = ref.replace("\n", " ")
            refsout.write(ref + "\n")

            hyp = hyp.replace("\n", " ")
            hypsout.write(hyp + "\n")


