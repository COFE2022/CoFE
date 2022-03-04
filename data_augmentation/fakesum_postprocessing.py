import argparse
import json

import pandas as pd
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, )
    parser.add_argument('--dataset_config_name', type=str, nargs='?')
    parser.add_argument('--data_dir', type=str, nargs='?')
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
        "--split",
        type=str,
        nargs='?',
        default="train",
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument('--generated_sum', type=str)
    parser.add_argument('--output', type=str, default="negative_train_data.csv")

    args = parser.parse_args()
    if args.split is None:
        args.split = 'train'
    source_col = args.text_column
    target_col = args.summary_column

    with open(args.generated_sum, "r") as input_file:
        res = json.load(input_file)

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, data_dir=args.data_dir)


    def add_columns(example, index):
        example.update({"order_num": index})
        return example


    dataset = dataset.map(add_columns, with_indices=True)

    t = dataset[args.split]

    df = t.to_pandas()
    # df2 = pd.DataFrame(data=res,columns=["generated_article","doc","order_num"])
    df2 = pd.DataFrame(data=res, columns=[f"generated_{target_col}", f"{target_col}_tokened", "order_num"])
    df = df.rename(columns={f"{target_col}": f"true_{target_col}"})

    df['order_num'] = df['order_num'].apply(int)
    df2['order_num'] = df2['order_num'].apply(int)
    df3 = df.merge(df2, on="order_num")
    df3 = df3.drop(columns=[f"{target_col}_tokened"])
    df3 = df3.rename(columns={f"generated_{target_col}": f"{target_col}"})
    df3 = df3.drop_duplicates("order_num")
    df3 = df3.sort_values("order_num")
    df3.to_csv(args.output, index=False)
