import argparse
import csv
import json
import pathlib
import pickle
from collections import defaultdict
from pathlib import Path

import datasets
import pandas as pd

from COFE.utils import summarization_name_mapping


def main():
    pass


if __name__ == '__main__':
    main()
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg_file_path', type=str, help='file path to the generated negative csv')
    parser.add_argument('--source_column', type=str, help='source column in header')
    parser.add_argument('--target_column', type=str, help='target column in header')
    parser.add_argument('--dataset_name', type=str, help='dataset name for loading "load_dataset" ', required=True)
    parser.add_argument('--dataset_config_name', type=str, help='dataset name for loading "load_dataset"', default=None)
    parser.add_argument('--cliff_pos_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    HF_data = datasets.load_dataset(args.dataset_name,
                                    args.dataset_config_name)
    column_names = HF_data[args.split].column_names
    HF_data_pd = {k: HF_data[k].to_pandas() for k, d in HF_data.items()}
    neg_df = pd.read_csv(args.neg_file_path)

    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.source_column is None:
        source_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        source_column = args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{args.source_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.target_column is None:
        target_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        target_column = args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{args.target_column}' needs to be one of: {', '.join(column_names)}"
            )

    hf_train_df = HF_data_pd[args.split]
    neg_df.reset_index()
    hf_train_df.reset_index()
    neg_df['neg_index'] = neg_df.index
    hf_train_df['hf_index'] = hf_train_df.index
    neg_df = neg_df.drop_duplicates(subset=['document'])
    hf_train_df = hf_train_df.drop_duplicates(subset=[source_column])
    df_mapping = pd.merge(neg_df, HF_data_pd[args.split], left_on='document', right_on=source_column)
    df_mapping = df_mapping.sort_values(by='hf_index').reset_index(drop=True)
    df_index_mapping = df_mapping.groupby('hf_index').agg({"neg_index": lambda x: list(x)}).reset_index()
    # result = df_index_mapping.to_dict()['neg_index']
    # result = {int(k): list(int(i) for i in v) for k, v in result.items()}
    result = defaultdict(list)
    for pair in df_mapping[['hf_index', 'neg_index']].iterrows():
        a = pair[-1].hf_index
        b = pair[-1].neg_index
        a = int(a)
        b = int(b)
        result[a].append(b)
    result = dict(result)
    save_path = Path(Path.cwd(), 'mapping_files', f"{args.dataset_name}_mapping_file.pickle")
    with open(save_path, 'wb') as outfile:
        pickle.dump(result, outfile)
    save_path2 = Path(Path.cwd(), 'mapping_files', f"{args.dataset_name}_mapping_file.txt")
    with open(save_path2, 'w') as outfile:
        json.dump(result, outfile)

    # sanity check
    for k, v in result.items():
        for i in v:
            a = HF_data[args.split][k]['document']
            b = neg_df.loc[neg_df.neg_index == int(i)].document[i]
            if not a == b:
                print(k)
                
    
