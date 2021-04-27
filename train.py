import argparse
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger as tb

from dataset import ImageRestorationDataModule
from module import Image2ImageModule
from net import ImageRestorationModel
import os

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


def get_module():
    model = ImageRestorationModel(3, 3)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-4)
    scheduler = sched.CosineAnnealingWarmRestarts(
        optimizer, 150)
    criterion = nn.L1Loss()
    return Image2ImageModule(model, optimizer, scheduler, criterion)


def train(args, parser):
    args, parser = train_args(parser)
    tb_logger = tb("..", "experiments", version=args.experiment_name)
    trainer = Trainer(
        gpus=args.gpu,
        logger=tb_logger,
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer.fit(get_module(), datamodule=ImageRestorationDataModule(args))
