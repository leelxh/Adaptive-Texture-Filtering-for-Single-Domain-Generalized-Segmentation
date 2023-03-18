from pytorch_lightning.loggers import CometLogger#, TensorboardLogger
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import random
from training_smooth import Smoothing
from datasets.smoothing_dataset import GTA5Dataset
import argparse 
import time


def main():
    parser = argparse.ArgumentParser(description="Yet another code-base for ytmt.")
    parser.add_argument("--name")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--max_epoch", default=20, type=int)
    parser.add_argument("--proj_name", default="dejpeg")
    parser.add_argument("--inet", default='smoothing_nafnet')
    parser.add_argument("--logdir", default="results")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
                "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--device_type", default='gpu', help="If 'gpu' passed, will train on gpu(s), if 'tpu' passed, will train on Google's TPU.")
    parser.add_argument("--logger", default=True)
    args = parser.parse_args()
    config = {"base_lr": args.base_lr, "max_epoch": args.max_epoch, "batch_size": args.bs, 
            "eps": 1e-2, "logdir" : args.logdir.rstrip('/\\') + '/' + args.name,
            "inet" : args.inet, "project_name" : args.proj_name,
            'name' : args.name + str(time.time()),
            "data_dir" : args.data_dir}

    os.makedirs(config["logdir"], exist_ok=True)
    sets = GTA5Dataset(config["data_dir"], size=24500)
    train_set, val_set = random_split(sets, [24490, 10], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], num_workers=10, drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=16, drop_last=False, shuffle=False, pin_memory=True, persistent_workers=True)

    # model
    ref_model = Smoothing(config)
    # training
    logger = True


    #trainer = pl.Trainer(gpus=None)
    if args.device_type == "gpu":
        trainer = pl.Trainer(accelerator="gpu", devices='auto', max_epochs=config['max_epoch'], 
                        logger=logger,
                        num_sanity_val_steps=2,  resume_from_checkpoint=args.resume, callbacks=[pl.callbacks.ModelCheckpoint(dirpath=args.proj_name + "/" + args.name + "/", save_top_k=2, monitor="step", mode="max"), pl.callbacks.RichProgressBar()])
    elif args.device_type == "tpu":  
        trainer = pl.Trainer(accelerator="tpu", devices=8, max_epochs=config['max_epoch'], 
                        logger=logger,
                        num_sanity_val_steps=0, resume_from_checkpoint=args.resume)



    trainer.fit(ref_model, train_loader, val_loader)  

if __name__ == "__main__":
    main()
