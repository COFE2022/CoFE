torchrun --nproc_per_node="number of cards " run_summarization.py \
  --model_name_or_path facebook/bart-large \
  --do_predict \
  --dataset_name xsum \
  --output_dir "where you store your model file/should same with finetune script" \
  --overwrite_output_dir \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --num_beams 6 \
  --predict_with_generate
