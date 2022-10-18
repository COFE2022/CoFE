# MLE baseline
This is a baseline for the MLE task. 
Modified from the huggingface example [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py)
Could be a good starting point for your own hyperparameter.

I use deepspeed ZeR03 mode and fp16 to accelerate the training, but it is still a bit of slower than the original fairseq implementation.(they use dynamic batch size and a lot of pre-processing optimization)