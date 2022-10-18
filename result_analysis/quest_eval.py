import argparse
from pathlib import Path

from questeval.questeval_metric import QuestEval

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--keep', type=int, default=None)
parser.add_argument('--source_column', type=str, default='srcs')
parser.add_argument('--target_column', type=str, default='labels')
parser.add_argument('--predict_column', type=str, default='preds')
parser.add_argument('--log_dir', type=str, default=None)
args = parser.parse_args()
questeval = QuestEval(no_cuda=False, use_cache=True, task='summarization', do_weighter=True, log_dir=args.log_dir)
result_dir = Path(args.result_path)
result_file = (result_dir / 'res.csv')
print('unweighted')

res = pd.read_csv(result_file.as_posix())

print(res.columns)

preds = res[args.predict_column].tolist()

labels = res[args.target_column].tolist()

srcs = res[args.source_column].tolist()

keep = args.keep

rouge_metric = load_metric("rouge")
result = rouge_metric.compute(predictions=preds, references=labels, use_stemmer=True)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
result = {k: round(v, 2) for k, v in result.items()}
result = [score for name, score in result.items()]
print(result)

preds = preds[:keep]
preds = [str(i) for i in preds]
srcs = srcs[:keep]
srcs = [str(i) for i in srcs]
labels = labels[:keep]
labels = [str(i) for i in labels]

labels = [[s] for s in labels]


score = questeval.corpus_questeval(
    hypothesis=preds,
    sources=srcs,
    list_references=labels,
    batch_size=128,
)

print(score['corpus_score'])
