import random
import os
from PIL import Image
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A
import torch
import numpy as np
import torch.nn.functional as nnf
from albumentations.pytorch import ToTensorV2


class ImageDataset(Dataset):
    """
    Data type can be "folder", "folders", and "image"
    """
    def __init__(self, path, data_type="folders", return_tensors=True):
        if data_type == "folders":
            self.paths = [
                os.path.join(path, folder, img_name)
                for folder in sorted(os.listdir(path))
                for img_name in sorted(os.listdir(os.path.join(path, folder))) 
            ]
        elif data_type == 'folder':
            self.paths = [
                os.path.join(path, img_name)
                for img_name in sorted(os.listdir(path)) 
            ]
        elif data_type == "image":
            self.paths = [path]
        
        self.return_tensors = return_tensors
        if return_tensors:
            self.transforms = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]))
        return self.transforms(image=img)["image"] if self.return_tensors else img


class ImageRestorationDataset(Dataset):
    def __init__(self, input_path, gt_path, patch_size, center_crop, scale_factor=4, data_type="folders"):
        self.input_dataset = ImageDataset(input_path, data_type, False)
        self.gt_dataset = ImageDataset(gt_path, data_type, False)
        self.transforms = A.Compose([
            A.CenterCrop(patch_size, patch_size) 
            if center_crop else 
            A.RandomCrop(patch_size, patch_size, always_apply=True),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={
            "gt":"image"
        })
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, idx):
        input = self.input_dataset[idx]
        gt = self.gt_dataset[idx]
        input = nnf.interpolate(
            torch.from_numpy(input).unsqueeze(0).permute(0,3,1,2), size=gt.shape[:-1], 
            mode='nearest'
        ).squeeze(0).permute(1,2,0).numpy()
        t = self.transforms(image=input, gt=gt)
        input = t["image"]
        gt = t["gt"]
        input = nnf.interpolate(
            input.unsqueeze(0), size=(self.patch_size // self.scale_factor, self.patch_size // self.scale_factor), 
            mode='nearest'
        ).squeeze(0)
        return input, gt


class ImageRestorationDataModule(LightningDataModule):
    def __init__(self, options):
        super().__init__()
        self.opt = options
        self.num_workers = self.opt.num_workers
        if self.opt.mode == 'train':
            self.train_dataset = ImageRestorationDataset(
                self.opt.train_input_path, self.opt.train_gt_path, self.opt.patch_size, False, self.opt.scale_factor, "folders"
            )
            self.valid_dataset = ImageRestorationDataset(
                self.opt.valid_input_path, self.opt.valid_gt_path, self.opt.patch_size, False, self.opt.scale_factor, "folders"
            )
        elif self.opt.mode == 'test':
            if self.opt.has_gt:
                self.test_dataset = ImageRestorationDataset(
                    self.opt.test_input_path, self.opt.test_gt_path, self.opt.patch_size, False, self.opt.scale_factor, self.opt.data_type
                )
            else:
                self.test_dataset = ImageDataset(
                    self.opt.test_input_path, self.opt.data_type, True
                )         

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.opt.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.opt.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 1,
            num_workers=self.num_workers
        )
