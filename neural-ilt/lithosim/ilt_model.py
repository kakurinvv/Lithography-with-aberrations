import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from lt_simulator import LTSimulator

# -------- 2. Model --------
class LithoZernikeRegressor(LightningModule):
    def __init__(self, model_name='efficientnet_b0', num_zernike=6, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = get_backbone(model_name, in_chans=2, num_classes=num_zernike)

        self.sim = LTSimulator()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        imgs, z_true = batch
        z_preds = self(imgs)
        litho_imgs, aberr_imgs = torch.split(imgs, 1, dim=1) # [B, 2, H, W] -> [B, (1, 1), H, W]
        restored_imgs = self.sim.run_lithosim(aberr_imgs, zernike_coeffs=z_preds)
        # loss = torch.nn.functional.mse_loss(preds, coeffs)
        loss = torch.nn.functional.l1_loss(restored_imgs, litho_imgs) * 1e3
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, z_true = batch
        z_preds = self(imgs)
        litho_imgs, aberr_imgs = torch.split(imgs, 1, dim=1) # [B, 2, H, W] -> [B, (1, 1), H, W]
        restored_imgs = self.sim.run_lithosim(aberr_imgs, zernike_coeffs=z_preds)
        # loss = torch.nn.functional.mse_loss(preds, coeffs)
        loss = torch.nn.functional.l1_loss(restored_imgs, litho_imgs) * 1e3
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def litho_loss(pred, true, threshold, margin: float = 0.1):
    sign = torch.sign(true - threshold)
    return torch.clip((true-pred)*sign + margin, min=0.).mean() / torch.std(true)

def get_backbone(model_name, in_chans=2, pretrained=True, num_classes=20):
    if model_name.startswith('resnet'):
        backbone = getattr(models, model_name)(pretrained=pretrained)
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
        backbone = getattr(models, model_name)(pretrained=pretrained)
        # Первый слой EfficientNet называется "features[0][0]"
        backbone.features[0][0] = torch.nn.Conv2d(
            in_chans, backbone.features[0][0].out_channels,
            kernel_size=3, stride=2, padding=1, bias=False)
        num_features = backbone.classifier[1].in_features
        backbone.classifier[1] = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith('convnext'):
        backbone = getattr(models, model_name)(pretrained=pretrained)
        backbone.features[0][0] = torch.nn.Conv2d(
            in_chans, backbone.features[0][0].out_channels,
            kernel_size=4, stride=4)
        num_features = backbone.classifier[2].in_features
        backbone.classifier[2] = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith('swin'):
        backbone = getattr(models, model_name)(pretrained=pretrained)
        backbone.features[0][0] = torch.nn.Conv2d(
            in_chans, backbone.features[0][0].out_channels,
            kernel_size=4, stride=4)
        num_features = backbone.head.in_features
        backbone.head = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith('vit'):
        backbone = getattr(models, model_name)(pretrained=pretrained)
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