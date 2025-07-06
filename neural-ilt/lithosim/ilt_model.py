import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint as cp
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from lt_simulator import LTSimulator

# -------- 2. Model --------
@torch.compile#(backend='cudagraphs')
class LithoZernikeRegressor(LightningModule):
    def __init__(self, model_name='efficientnet_b0', num_zernike=6, lr=1e-3, checkpointing: bool = True):
        super().__init__()
        self.save_hyperparameters()
        
        self.sim = LTSimulator(checkpointing=checkpointing)
        self.backbone = get_backbone(model_name, in_chans=2, num_classes=num_zernike)

        self.loss_func = torch.nn.functional.l1_loss
        self.checkpointing = checkpointing


    def forward(self, x):
        if self.checkpointing:
            def run_func(input):
                return self.backbone(input)
            return cp.checkpoint(run_func, x, use_reentrant=False)
        else:
            return self.backbone(x)

    def training_step(self, batch, batch_idx):
        imgs, z_true = batch
        litho_aberr_imgs = imgs[:, 1:] # skip design images
        z_preds = self(litho_aberr_imgs)
        design_imgs, litho_imgs, aberr_imgs = torch.split(imgs, 1, dim=1) # [B, 3, H, W] -> [B, (1, 1, 1), H, W]
        
        modelled_imgs = self.sim.run_lithosim(design_imgs, zernike_coeffs=z_preds)
        loss = self.loss_func(modelled_imgs, aberr_imgs, reduction='mean')

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, z_true = batch
        litho_aberr_imgs = imgs[:, 1:] # skip design images
        z_preds = self(litho_aberr_imgs)
        design_imgs, litho_imgs, aberr_imgs = torch.split(imgs, 1, dim=1) # [B, 3, H, W] -> [B, (1, 1, 1), H, W]
        
        modelled_imgs = self.sim.run_lithosim(design_imgs, zernike_coeffs=z_preds)
        loss = self.loss_func(modelled_imgs, aberr_imgs, reduction='mean')

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    
    def on_after_backward(self):
        # self.parameters() только с requires_grad
        grads = []
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.detach().abs().mean().item()
                self.log(f'grad_mean_{n}', grad_norm, prog_bar=False, logger=True)
                grads.append(grad_norm)
        # Можно вывести среднее по всем слоям:
        if grads:
            mean_grad = sum(grads) / len(grads)
            self.log('mean_grad', mean_grad, prog_bar=True, logger=True)


# @torch.compile
def litho_loss(pred, true, threshold=0.225, margin: float = 0.1, power: float = 2.):
    sign = torch.sign(true - threshold)
    return (
        torch.clip((true-pred)*sign + margin, min=0.) ** power
    ).sum(dim=[-1,-2]).mean()# / torch.std(true)

# @torch.compile
def get_backbone(model_name: str, in_chans=2, pretrained=True, num_classes=20):
    backbone = getattr(models, model_name)(pretrained=pretrained)

    if model_name.startswith('resnet'):
        # Меняем первый слой под два канала
        backbone.conv1 = torch.nn.Conv2d(
            in_chans, backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False)
        num_features = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith('efficientnet'):
        # Первый слой EfficientNet называется "features[0][0]"
        backbone.features[0][0] = torch.nn.Conv2d(
            in_chans, backbone.features[0][0].out_channels,
            kernel_size=3, stride=2, padding=1, bias=False)
        num_features = backbone.classifier[1].in_features
        backbone.classifier[1] = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith("mobilenet"):
        backbone.features[0][0] = torch.nn.Conv2d(
            in_chans, backbone.features[0][0].out_channels,
            kernel_size=3, stride=2, padding=1, bias=False)
        num_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith('convnext'):
        backbone.features[0][0] = torch.nn.Conv2d(
            in_chans, backbone.features[0][0].out_channels,
            kernel_size=4, stride=4)
        num_features = backbone.classifier[2].in_features
        backbone.classifier[2] = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith('swin'):
        backbone.features[0][0] = torch.nn.Conv2d(
            in_chans, backbone.features[0][0].out_channels,
            kernel_size=4, stride=4)
        num_features = backbone.head.in_features
        backbone.head = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith('vit'):
        # vit expects 3 channels by default, patchify works on in_chans
        backbone.conv_proj = torch.nn.Conv2d(
            in_chans, backbone.conv_proj.out_channels,
            kernel_size=backbone.conv_proj.kernel_size,
            stride=backbone.conv_proj.stride,
            padding=backbone.conv_proj.padding,
            bias=False)
        num_features = backbone.heads.head.in_features
        backbone.heads.head = torch.nn.Linear(num_features, num_classes)
    else:
        raise NotImplementedError(f'Unknown model: {model_name}')
    return backbone