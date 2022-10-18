import argparse
from pathlib import Path

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import set_seed

from utils import mykey
from pl_cofe_model import CoFE_pl_model


def training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=-1,
        help='''Stop training once this number of epochs is reached. Disabled by default (None).
            If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
            To enable infinite training, set ``max_epochs = -1``.''',
    )

    parser.add_argument(
        '--num_devices',
        type=int,
        default=-1,
        help="""Number of devices the trainer uses per node."""
    )
    parser.add_argument(
        '--val_check_interval',
        type=float,
        default=1.0,
        help="""How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches.
                Default: ``1.0``."""
    )

    parser.add_argument(
        '--fp16',
        type=bool,
        default=False,
        help="""Whether to use fp16 (mixed) precision instead of
                        32-bit (default: False)"""
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        help="""    Supports different training strategies with aliases as
                        well custom strategies. Default: ``None``. (type:
                        Union[str, Strategy, null], default: null)"""
    )
    parser.add_argument(
        '--weight_file',
        type=str,
        default=None,
        help="""    do fit """
    )
    parser.add_argument(
        '--do_train',
        action='store_true',
        help="""    do fit """
    )
    parser.add_argument(
        '--do_eval',
        action='store_true',
        help="""    do fit """
    )

    parser.add_argument(
        '--do_predict',
        action='store_true',
        help="""    do_predict """
    )
    parser.add_argument(
        '--do_test',
        action='store_true',
        help="""    do test """
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint file.",
    )

    return parser


def main():
    global trainer
    parser = argparse.ArgumentParser()
    parser = training_args(parser)
    parser = CoFE_pl_model.add_model_specific_args(parser)
    argument = parser.parse_args()
    if argument.max_epochs is None and argument.max_steps is None:
        raise ValueError('max_epochs and max_steps can\'t be None at same time')
    precision = 16 if argument.fp16 else 32
    if type(argument.val_check_interval) == str:
        if '.' in argument.val_check_interval:
            argument.val_check_interval = float(argument.val_check_interval)
        else:
            argument.val_check_interval = int(argument.val_check_interval)
    model = CoFE_pl_model(argument)
    wandb.login(key=mykey)
    wandb_path = Path(argument.output_dir)
    wandb_logger = WandbLogger(project=f'COFE_XSUM_{argument.negative_weight}', save_dir=wandb_path.as_posix())
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_rouge1',
        dirpath=argument.output_dir,
        save_top_k=10,
        mode='max',
        save_last=True,
        filename='s2s-{epoch:02d}-{val_rouge1:.2f}',
        auto_insert_metric_name=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if argument.seed is not None:
        set_seed(argument.seed)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    if accelerator == 'gpu':
        trainer = Trainer(devices=argument.num_devices,
                          accelerator=accelerator,
                          strategy=argument.strategy,
                          default_root_dir=argument.output_dir,
                          logger=wandb_logger,
                          gradient_clip_val=argument.clip_norm,
                          precision=precision,
                          callbacks=[checkpoint_callback, lr_monitor],
                          val_check_interval=argument.val_check_interval,
                          max_steps=argument.max_steps,
                          max_epochs=argument.max_epochs,
                          accumulate_grad_batches=argument.gradient_accumulation_steps,
                          )
    elif accelerator == 'cpu':
        trainer = Trainer(
            accelerator=accelerator,
            default_root_dir=argument.output_dir,
            logger=wandb_logger,
            gradient_clip_val=argument.clip_norm,
            callbacks=[checkpoint_callback, lr_monitor],
            val_check_interval=argument.val_check_interval,
            max_steps=argument.max_steps,
            max_epochs=argument.max_epochs,
            accumulate_grad_batches=argument.gradient_accumulation_steps,
            num_sanity_val_steps=2,
        )

    # wandb_logger.watch(model)
    if argument.do_train:
        trainer.fit(model, ckpt_path=argument.resume_from_checkpoint)
        print(checkpoint_callback.best_model_path)
    if argument.do_eval:
        trainer.validate(
            model,
            ckpt_path=argument.resume_from_checkpoint
        )
    if argument.do_test:
        trainer.test(
            model,
            ckpt_path=argument.resume_from_checkpoint
        )
    if argument.do_predict:
        trainer.predict(
            model,
            ckpt_path=argument.resume_from_checkpoint,
            dataloaders=model.test_dataloader()
        )


if __name__ == '__main__':
    main()
