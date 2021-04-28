import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger as tb

from dataset import ImageRestorationDataModule
from module import Image2ImageModule
from net import ImageRestorationModel
import os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def train_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--batch_size", '-bs', type=int, default=4)
    parser.add_argument(
        "--epochs", type=int, default=4)
    parser.add_argument(
        "--experiment_name", "-exn", type=str,
        default=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    parser.add_argument(
        "--patch_size", '-ps', type=int, default=64)
    parser.add_argument(
        "--scale_factor", '-sf', type=int, default=4)


    parser.add_argument(
        "--train_input_path", '-tip', type=str, 
        default=os.path.realpath("dataset/GoPro/corrupted")
    )
    parser.add_argument(
        "--train_gt_path", '-tgp', type=str, 
        default=os.path.realpath("dataset/GoPro/gt")
    )
    parser.add_argument(
        "--valid_input_path", '-vip', type=str, 
        default=os.path.realpath("dataset/GoPro/corrupted")
    )
    parser.add_argument(
        "--valid_gt_path", '-vgp', type=str, 
        default=os.path.realpath("dataset/GoPro/gt")
    )
        
    args, _ = parser.parse_known_args()
    return args, parser


def get_module(args):
    model = ImageRestorationModel(3, 3)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-3)
    scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5),
        'monitor': "PSNR_epoch",
    }
    criterion = nn.L1Loss()
    return Image2ImageModule(model, optimizer, scheduler, criterion, args)


def train(args, parser):
    args, parser = train_args(parser)
    tb_logger = tb("experiments", version=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("experiments", args.experiment_name),
        save_top_k=1,
        monitor="PSNR",
        mode="max"
    )
    trainer = Trainer(
        gpus=args.gpu,
        logger=tb_logger,
        limit_val_batches=0,
        # limit_train_batches=2,
        num_sanity_val_steps=0,
        deterministic=True,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint= args.pretrained_path if args.resume else None,
    )
    trainer.fit(get_module(args), datamodule=ImageRestorationDataModule(args))
