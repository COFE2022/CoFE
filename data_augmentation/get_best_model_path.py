from os import listdir
from os.path import isfile, join
import re
import argparse
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--root",type=str,required=True)
args = parser.parse_args()


mypath = args.root
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

rouge1s = [re.findall(r"rouge1=[-+]?([0-9]*.[0-9]+|[0-9]+)",i)for i in onlyfiles]
rouge1s =  [str(i[-1]) for i in rouge1s if len(i)>0]

max_value = max(rouge1s)
best_model = [name for name in onlyfiles if str(max_value) in name][0]
print(pathlib.Path(mypath,best_model).as_posix())