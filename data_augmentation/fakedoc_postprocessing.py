import argparse
import json

import pandas as pd
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, )
    parser.add_argument('--dataset_config_name', type=str, default=None, nargs='?')
    parser.add_argument('--data_dir',type=str,default=None, nargs='?')
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
        nargs='?',
        type=str,
        default="train",
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )

    parser.add_argument('--generated_doc', type=str)
    parser.add_argument('--output', type=str, default="btrans_data.csv")


    args = parser.parse_args()
    print(args)
    if args.split==None:
        args.split='train'

    with open(args.generated_doc,"r") as input_file:
        res = json.load(input_file)

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, data_dir=args.data_dir)


    def add_columns(example, index):
        example.update({"order_num": index})
        return example


    dataset = dataset.map(add_columns, with_indices=True)

    t = dataset[args.split]

    df = t.to_pandas()
    # df2 = pd.DataFrame(data=res,columns=["generated_article","doc","order_num"])
    df2 = pd.DataFrame(data=res, columns=[f"generated_{args.text_column}", f"{args.text_column}_tokened", "order_num"])
    df = df.rename(columns={f"{args.text_column}": f"true_{args.text_column}"})



    df['order_num'] = df['order_num'].apply(int)
    df2['order_num'] = df2['order_num'].apply(int)
    df3 = df.merge(df2, on="order_num")
    df3 = df3.drop(columns=[f"{args.text_column}_tokened"])
    df3 = df3.rename(columns={f"generated_{args.text_column}": f"{args.text_column}"})
    df3 = df3.drop_duplicates("order_num")
    df3 = df3.sort_values("order_num")
    df3.to_csv(args.output,index=False)





