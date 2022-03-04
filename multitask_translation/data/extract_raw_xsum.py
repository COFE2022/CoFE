# import datasets
# x = datasets.load_dataset('xsum')
# with open('train.source','w') as inputfile1:
#     with open('train.target','w') as inputfile2:
#         for e in x["train"]:
#             source = e['document']
#             source = source.replace('\n',' ')
#             target = e['summary']
#             target = target.replace('\n',' ')
#
#             source = '[ent] '+ source
#             inputfile1.write(source+"\n")
#             inputfile2.write(target+"\n")
#
# with open('val.source', 'w') as inputfile1:
#     with open('val.target', 'w') as inputfile2:
#         for e in x["validation"]:
#             source = e['document']
#             source = source.replace('\n',' ')
#             target = e['summary']
#             target = target.replace('\n',' ')
#
#             source = '[ent] ' + source
#             inputfile1.write(source + "\n")
#             inputfile2.write(target + "\n")
#
# with open('test.source', 'w') as inputfile1:
#     with open('test.target', 'w') as inputfile2:
#         for e in x["test"]:
#             source = e['document']
#             source = source.replace('\n',' ')
#             target = e['summary']
#             target = target.replace('\n',' ')
#
#             source = '[ent] ' + source
#             inputfile1.write(source + "\n")
#             inputfile2.write(target + "\n")
import argparse
import pandas as pd
import pathlib
import datasets
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, )
parser.add_argument('--dataset_config_name', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--neg_csv', required=True, type=str)
parser.add_argument('--keep_neg', default=1, type=float)

parser.add_argument('--pos_out_dir', type=str, required=True)
parser.add_argument('--neg_out_dir', type=str, required=True)

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


args = parser.parse_args()
source_col = args.text_column
target_col = args.summary_column
main_dataset = datasets.load_dataset(args.dataset_name, args.dataset_config_name, args.data_dir)

print(main_dataset)
def add_index(example, index):
    example.update({"order_num": index})
    example.update({"task": 'sum'})
    return example


main_dataset = main_dataset.map(add_index, with_indices=True)
t = main_dataset["train"]
df_train = t.to_pandas()
test_set = main_dataset["test"]
df_test = test_set.to_pandas()
vaL_set = main_dataset["validation"]
df_val = vaL_set.to_pandas()




df_neg = pd.read_csv(args.neg_csv)
neg_keep_size = min(int(df_train.shape[0]*args.keep_neg), df_neg.shape[0])

# df_neg = df_neg[df_train.columns]
print(df_train)
print(df_neg)
df_neg = df_neg.sample(neg_keep_size,random_state=1)

print(df_train.columns,df_neg.columns)

df_neg['nli'] = 'con'

df_train['nli'] = 'ent'


def Add_prefix(nli, summary):
    prefix = f"[{nli}] "
    return prefix + summary

df_train[source_col] = df_train.apply(lambda row: Add_prefix(row['nli'], str(row[source_col])), axis=1)
df_neg[source_col] =  df_neg.apply(lambda row: Add_prefix(row['nli'], str(row[source_col])), axis=1)

df_v_con = df_val.copy()
df_v_con['nli'] = 'con'
df_v_ent = df_val.copy()
df_v_ent['nli'] = 'ent'
df_v_con[source_col] = df_v_con.apply(lambda row: Add_prefix(row['nli'], row[source_col]), axis=1)
df_v_ent[source_col] = df_v_ent.apply(lambda row: Add_prefix(row['nli'], row[source_col]), axis=1)

df_test_con = df_test.copy()
df_test_con['nli'] = 'con'
df_test_ent = df_test.copy()
df_test_ent['nli'] = 'ent'
df_test_con[source_col] = df_test_con.apply(lambda row: Add_prefix(row['nli'], row[source_col]), axis=1)
df_test_ent[source_col] = df_test_ent.apply(lambda row: Add_prefix(row['nli'], row[source_col]), axis=1)




df_train = df_train.rename(columns={source_col: "source", target_col: "target"})
df_neg = df_neg.rename(columns={source_col: "source", target_col: "target"})
df_v_con = df_v_con.rename(columns={source_col: "source", target_col: "target"})
df_v_ent = df_v_ent.rename(columns={source_col: "source", target_col: "target"})
df_test_con = df_test_con.rename(columns={source_col: "source", target_col: "target"})
df_test_ent = df_test_ent.rename(columns={source_col: "source", target_col: "target"})

print(df_train)
print(df_neg)
print(df_test_ent)
print(df_test_ent)

with open(pathlib.Path(args.pos_out_dir, "train.source").as_posix(),'w') as trainfile:
    with open(pathlib.Path(args.pos_out_dir, "train.target").as_posix(),'w') as targetfile:
        source = df_train.source.tolist()
        target = df_train.target.tolist()
        for i in source:
            sent = i.replace('\n', ' ')
            trainfile.write(sent + '\n')

        for i in target:
            sent = i.replace('\n', ' ')
            targetfile.write(sent + '\n')

with open(pathlib.Path(args.pos_out_dir, "valid.source").as_posix(),'w') as trainfile:
    with open(pathlib.Path(args.pos_out_dir, "valid.target").as_posix(),'w') as targetfile:
        source = df_v_ent.source.tolist()
        target = df_v_ent.target.tolist()
        for i in source:
            sent = i.replace('\n', ' ')
            trainfile.write(sent + '\n')

        for i in target:
            sent = i.replace('\n', ' ')
            targetfile.write(sent + '\n')

with open(pathlib.Path(args.pos_out_dir, "test.source").as_posix(),'w') as trainfile:
    with open(pathlib.Path(args.pos_out_dir, "test.target").as_posix(),'w') as targetfile:
        source = df_test_ent.source.tolist()
        target = df_test_ent.target.tolist()
        for i in source:
            sent = i.replace('\n', ' ')
            trainfile.write(sent + '\n')

        for i in target:
            sent = i.replace('\n', ' ')
            targetfile.write(sent + '\n')

with open(pathlib.Path(args.neg_out_dir, "train.source").as_posix(),'w') as trainfile:
    with open(pathlib.Path(args.neg_out_dir, "train.target").as_posix(),'w') as targetfile:
        source = df_neg.source.tolist()
        target = df_neg.target.tolist()
        for i in source:
            sent = i.replace('\n', ' ')
            trainfile.write(sent + '\n')

        for i in target:
            sent = i.replace('\n', ' ')
            targetfile.write(sent + '\n')
