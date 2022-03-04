import tqdm
from datasets import load_metric
from newsroom.analyze import Fragments
from questeval.questeval_metric import QuestEval
# questeval = QuestEval(no_cuda=False)
questeval = QuestEval(no_cuda=False,use_cache=True,task='summarization', do_weighter=True)
# xsum
hypo_file = './outdir/xsum_nli_weight/formatted-test.txt'
source_file = '/scratch/tw2112/codes/ablation/xsum_weight/pos_raw/test.source'
label_file = '/scratch/tw2112/codes/ablation/xsum_weight/pos_raw/test.target'

# giga
# source_file = '/scratch/tw2112/codes/ablation/giga_weight/pos_raw/test.source'
# hypo_file = './outdir/giga_weight_nli/formatted-test.txt'
# label_file = '/scratch/tw2112/codes/ablation/giga_weight/pos_raw/test.target'

# wiki
# source_file = '/scratch/tw2112/codes/ablation/wiki_weight/pos_raw/test.source'
# hypo_file = 'outdir/wiki_weight_nli/formatted-test.txt'
# label_file  = '/scratch/tw2112/codes/ablation/wiki_weight/pos_raw/test.target'

import  pandas as pd

keep = 20000
with open(label_file,'r') as reffile:
    refs = []
    for i in reffile:
        i = i.replace("[ent] ", "")
        refs.append(i.strip())


with open(hypo_file,'r') as outputfile:
    preds = []
    for i in outputfile:
        i = i.replace("[ent] ", "")
        preds.append(i.strip())

with open(source_file,'r') as outputfile:
    srcs = []
    for i in outputfile:
        i = i.replace("[ent] ", "")
        srcs.append(i.strip())

rouge_metric = load_metric("rouge")
result = rouge_metric.compute(predictions=preds, references=refs, use_stemmer=True)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
result = {k: round(v, 2) for k, v in result.items()}
result = [score for name, score in result.items()]
print(result)


dfres = pd.DataFrame(columns=['Coverage','Density'])
docs = srcs
preds = preds
docs = [str(doc) for doc in docs]
preds = [str(pred) for pred in preds]
coverages = []
densities = []
for i in tqdm.tqdm(range(len(docs))):
    doc = docs[i]
    pred = preds[i]
    fragments = Fragments(pred,doc)

    dfres.loc[i]=[fragments.coverage(), fragments.density()]

# print(dfres)
print(f"{dfres.Coverage.mean():.4f}")
print(f"{dfres.Density.mean():.4f}")

preds = preds[:keep]
srcs = srcs[:keep]
refs = refs[:keep]

score = questeval.corpus_questeval(
    hypothesis=preds,
    sources=srcs,
    list_references=refs
)

print(score['corpus_score'])






