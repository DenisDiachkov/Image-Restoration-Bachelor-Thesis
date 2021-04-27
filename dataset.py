import random
from PIL import Image
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageDataset(Dataset):
    """
    Data type can be "folder", "folders", and "image"
    """
    def __init__(self, path, data_type="folders"):
        if data_type == "folders":
            self.paths = [
                os.path.join(path, folder, img_name)
                for img_name in sorted(os.listdir(os.path.join(path, folder))) 
                for folder in sorted(os.listdir(path))
            ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.paths[idx]))


class ImageRestorationDataset(Dataset):
    def __init__(self, input_path, gt_path, patch_size, center_crop):
        self.input_dataset = ImageDataset(input_path)
        self.gt_dataset = ImageDataset(gt_path)
        self.transforms = A.Compose([
            A.CenterCrop(patch_size, patch_size) 
            if center_crop else 
            A.RandomCrop(patch_size, patch_size, always_apply=True),
            
            A.Normalize(),
            ToTensorV2(),
        ], additional_targets={
            "gt":"image"
        })
        
    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, idx):
        input = self.input_dataset[idx]
        gt = self.input_dataset[idx]
        t = self.transforms(image=input, gt=gt)
        input = t["image"]
        gt = t["gt"]
        return input, gt


class ImageRestorationDataModule(LightningDataModule):
    def __init__(self, options):
        super().__init__()
        self.opt = options
        self.num_workers = self.opt.num_workers
        if self.opt.mode == 'train':
            self.train_dataset = ImageRestorationDataset(
                self.opt.train_input_path, self.opt.train_gt_path, self.opt.patch_size, True
            )
            self.valid_dataset = ImageRestorationDataset(
                self.opt.valid_input_path, self.opt.valid_gt_path, self.opt.patch_size, False
            )
        elif self.opt.mode == 'test':
            self.test_dataset = ImageRestorationDataset(
                self.opt.test_input_path, self.opt.test_gt_path, -1, False
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
            self.test_dataset, batch_size=self.opt.batch_size,
            num_workers=self.num_workers
        )
