from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import psnr, ssim
import torch
from torchvision.transforms import ToPILImage   
import os
import utils


class Image2ImageModule(LightningModule):
    def __init__(self, model, optimizer, scheduler, criterion, options):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.opt = options

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = self.criterion(p, y)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('PSNR', psnr(p, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('SSIM', ssim(p, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        if self.opt.has_gt:
            x, y = batch
            p = self(x)
            loss = self.criterion(p, y)
            self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PSNR', psnr(p, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('SSIM', ssim(p, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            p = self(batch)

        if not os.path.exists(self.opt.output_path):
            os.makedirs(self.opt.output_path)
        savepath = os.path.join(self.opt.output_path, f"{len(os.listdir(self.opt.output_path)):03}.png")
        img = torch.clamp(utils.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(p[0]), 0.0, 1.0).cpu()
        ToPILImage()(img).save(savepath)

    def configure_optimizers(self):
        if self.optimizer is None:
            return None
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]
        if self.scheduler is None:
            return self.optimizer
        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]
        return self.optimizer, self.scheduler
