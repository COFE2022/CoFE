import pandas as pd
dfres = pd.read_csv('xsum_neg.csv')
source = dfres.document.tolist()
summary = dfres.summary.tolist()
print(len(source), len(summary))
with open('train.source','w') as outfile:

    for i in source:
        i = str(i)
        i = i.replace('\n',' ')
        outfile.write(i.strip()+"\n")

with open('train.target','w') as outfile:

    for i in summary:
        i = str(i)
        i = i.replace('\n', ' ')
        outfile.write(i.strip()+"\n")