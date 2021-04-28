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
import utils
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def train_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False)
    
    parser.add_argument(
        "--patch_size", '-ps', type=int, default=64)
    parser.add_argument(
        "--scale_factor", '-sf', type=int, default=4)

    parser.add_argument(
        "--test_input_path", '-vip', type=str, 
        default=os.path.realpath("dataset/GoPro/corrupted/000/000.png")
    )
    parser.add_argument(
        "--test_gt_path", '-vgp', type=str, 
        default=os.path.realpath("dataset/GoPro/gt")
    )

    parser.add_argument(
        "--output_path", '-o', type=str, 
        default="Test_output"
    )
    
    parser.add_argument(
        "--data_type", '-dt', type=str, 
        default="image", choices=["image", "folder", "folders"]
    )

    parser.add_argument(
        "--has_gt", action="store_true"
    )

    args, _ = parser.parse_known_args()
    return args, parser


def get_module(args):
    model = ImageRestorationModel(3, 3)
    model.load_state_dict(utils.removeStateDictPrefix(torch.load(args.pretrained_path)['state_dict'], 6))
    return Image2ImageModule(model, None, None, None, args)


def test(args, parser):
    args, parser = train_args(parser)

    trainer = Trainer(
        gpus=args.gpu,
        deterministic=True,
        resume_from_checkpoint = args.pretrained_path
    )
    trainer.test(get_module(args), test_dataloaders=[
        ImageRestorationDataModule(args).test_dataloader()
    ])
