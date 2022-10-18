# Loss Truncation

- pip install -U git+https://github.com/ddkang/loss_dropper.git
- install the fairseq from [git repo](https://github.com/facebookresearch/fairseq)(not pypi), I suggest you to create a new conda env and install the fairseq there. 
- About the data preprocessing in fairseq, please follow this [link](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.summarization.md). In a nutshell, put source and target in two files separately, then convert them into binary files by fairseq's cli.
- Add --user-dir PATH(where the custom loss function is) to the fairseq-train command, it will automatically register the new function to fairseq
- Note that there are some noise in the dataset, filter out all the examples with empty target or source
```shell
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=4098
UPDATE_FREQ=4
BART_PATH=~/models/BART/bart.large/model.pt
CUDA_VISIBLE_DEVICES=2,3 fairseq-train ~/data/xsum/bin/ \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy_with_loss_trunc \
    --label-smoothing 0.1 \
    --dropc 0.6 \
    --min-count 200000 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --max-epoch 3 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
```