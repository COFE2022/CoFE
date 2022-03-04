set -e

DATA="/scratch/tw2112/codes/ablation/wiki_weight"


mkdir $DATA/nli_raw -p
mkdir $DATA/nli_bin -p

python extract_nli_data.py --out_dir $DATA/nli_raw


for SPLIT in train
do
  for LANG in source target
  do
    python  -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json  \
            --vocab-bpe vocab.bpe \
            --inputs "$DATA/nli_raw/$SPLIT.$LANG" \
            --outputs "$DATA/nli_raw/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
  done
done

fairseq-preprocess --source-lang source --target-lang target \
 --trainpref $DATA/nli_raw/train.bpe \
 --destdir $DATA/nli_bin\
 --workers 60 \
 --srcdict dict.txt \
 --tgtdict dict.txt




