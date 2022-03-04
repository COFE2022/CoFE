set -e
DATA=/scratch/tw2112/codes/ablation/xsum
for SPLIT in train val test
do
  for LANG in source target
  do
    python  -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json  \
            --vocab-bpe vocab.bpe \
            --inputs "$DATA/xsum_raw_label/$SPLIT.$LANG" \
            --outputs "$DATA/xsum_raw_label/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
  done
done

fairseq-preprocess --source-lang source --target-lang target \
 --trainpref $DATA/xsum_raw_label/train.bpe \
 --validpref $DATA/xsum_raw_label/val.bpe \
 --testpref $DATA/xsum_raw_label/test.bpe \
 --destdir $DATA/pos_binarized \
 --workers 60 \
 --srcdict dict.txt \
 --tgtdict dict.txt






for SPLIT in train
do
  for LANG in source target
  do
    python  -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json  \
            --vocab-bpe vocab.bpe \
            --inputs "$DATA/neg_raw/$SPLIT.$LANG" \
            --outputs "$DATA/neg_raw/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
  done
done

fairseq-preprocess --source-lang source --target-lang target \
 --trainpref $DATA/neg_raw/train.bpe \
 --destdir $DATA/neg_binarized \
 --workers 60 \
 --srcdict dict.txt \
 --tgtdict dict.txt