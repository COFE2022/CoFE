import argparse
import pathlib
from os import listdir
from os.path import isfile, join
import re

import pandas as pd
import editdistance

parser=argparse.ArgumentParser()
parser.add_argument("--root", default="data",type=str)
parser.add_argument("--fold", default=5, type=int)
parser.add_argument("--name", default="xsum", type=str)
parser.add_argument("--summary_col", default="summary", type=str)
parser.add_argument("--min_d", default=2, type=int)
args = parser.parse_args()

root_path = args.root
onlyfiles = [f for f in listdir(root_path) if isfile(join(root_path, f))]
pattern = f"({args.name}_fake_sum_[\s\S]*.csv)"
files = [re.findall(pattern,i) for i in onlyfiles]
files = [str(i[0]) for i in files if len(i)>0]
print(files)
dfs = []
for csvfile in files:
    csvfile = pathlib.Path(root_path,csvfile)
    df = pd.read_csv(csvfile)
    dfs.append(df)

dfres = pd.concat(dfs)
dfres = dfres.sort_values(by="order_num")
# dfres = dfres[dfres[f'true_{args.summary_col}']!=dfres[f'{args.summary_col}']]
dfres['dist'] = dfres.apply(lambda x:editdistance.eval(x[f'true_{args.summary_col}'],x[f'{args.summary_col}']) , axis=1)
print(dfres['dist'].quantile(0.1),dfres['dist'].quantile(0.2))
dfres = dfres[dfres['dist'] >= args.min_d]
dfres.to_csv(pathlib.Path(root_path,f"{args.name}_neg.csv"),index=False)
print(dfres)
# dfres['dist'].to_csv(pathlib.Path(root_path,f"{args.name}_distance.csv"),index=False)
res = []
