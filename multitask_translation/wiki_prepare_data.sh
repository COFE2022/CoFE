set -e
DATA="/scratch/tw2112/codes/ablation/wiki_weight"
mkdir $DATA/pos_raw -p
mkdir $DATA/neg_raw -p
mkdir $DATA/pos_bin -p
mkdir $DATA/neg_bin -p

NEG_CSV=/home/tw2112/codes/s2s/aux_with_neg_wiki/data/playwikihow_neg.csv

python data/extract_raw_xsum.py  --dataset_name wikihow --dataset_config_name sep --data_dir $SCRATCH/datas/wikihow --neg_csv $NEG_CSV --pos_out_dir $DATA/pos_raw --neg_out_dir $DATA/neg_raw --keep_neg 0.5 --text_column text --summary_column headline




#pos part
for SPLIT in train valid test
do
  for LANG in source target
  do
    python  -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json  \
            --vocab-bpe vocab.bpe \
            --inputs "$DATA/pos_raw/$SPLIT.$LANG" \
            --outputs "$DATA/pos_raw/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
  done
done

fairseq-preprocess --source-lang source --target-lang target \
 --trainpref $DATA/pos_raw/train.bpe \
 --validpref $DATA/pos_raw/valid.bpe \
 --testpref $DATA/pos_raw/test.bpe \
 --destdir $DATA/pos_bin \
 --workers 60 \
 --srcdict dict.txt \
 --tgtdict dict.txt


#neg part
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
 --destdir $DATA/neg_bin \
 --workers 60 \
 --srcdict dict.txt \
 --tgtdict dict.txt



