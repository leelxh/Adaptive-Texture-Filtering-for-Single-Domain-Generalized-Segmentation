import torch
import torch.nn as nn
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from models import arch
from losses import dejpeg_losses
from utils.image_utils import imsave, tensor2img
import torchmetrics
from timm.scheduler import CosineLRScheduler


class Smoothing(pl.LightningModule):
    def __init__(self, config : dict):
        super().__init__()
        self.config = config
        self.in_channels = 3
        self.model = arch.__dict__[self.config['inet']](self.in_channels, [3, 3]).to(self.device)
        self.fid_loss = nn.L1Loss()
        self.grad_loss = dejpeg_losses.SmoothingTVLoss()
        self.lr = config["base_lr"]
        self.psnr = torchmetrics.PeakSignalNoiseRatio()

    def forward(self, image, lamb):
        return self.model(image, lamb)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineLRScheduler(optimizer,
                              warmup_lr_init = self.lr / 100,
                              warmup_t = self.config["max_epoch"] // 20, lr_min = 1e-6,
                              t_initial = self.config["max_epoch"] - (self.config["max_epoch"] // 20))
        #return [optimizer], [{"scheduler":scheduler, "interval" : "epoch"}]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metric_to_track",
            },
        }
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

    def training_step(self, train_batch, batch_idx):
        input_img, label_img, _ = train_batch
        lamb = torch.rand(1, device="cuda") 
        lamb = 4 - 4 * (lamb * lamb)
        ret = self.forward(input_img, lamb)
        fid_loss = self.fid_loss(ret, input_img)
        grad_loss = 0.2 * self.grad_loss(ret, input_img, label_img, lamb)
        loss = grad_loss + fid_loss

        self.log("train_grad_loss", grad_loss, logger=True, on_epoch=True, prog_bar=True)
        self.log("train_fid_loss", fid_loss, logger=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, logger=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input_img, label_img, file_name = val_batch
        lamb = torch.rand(1, device="cuda") #.unsqueeze(0).unsqueeze(0)
        #lamb = torch.tensor([4], device='cuda')
        lamb = 4 - 4 * (lamb * lamb)
        ret = self.forward(input_img, lamb)
        fid_loss = self.fid_loss(ret, input_img)
        grad_loss = 0.2 * self.grad_loss(ret, input_img, label_img, lamb)
        loss = grad_loss + fid_loss

        self.log("val_loss", loss, logger=True, on_epoch=True)

        if self.global_rank == 0:
            img = tensor2img(ret)
            target = tensor2img(input_img)
            #print( "%s/result_%d_%.4f_%s.png" % (self.config["logdir"], self.current_epoch, lamb.item(), file_name[0].split('/')[-1]))
            imsave(img, "%s/result_%d_%.4f_%s.png" % (self.config["logdir"], self.current_epoch, lamb.item(), file_name[0].split('/')[-1]))
            imsave(target, "%s/target_%d_%s.png" % (self.config["logdir"], self.current_epoch, file_name[0].split('/')[-1]))


