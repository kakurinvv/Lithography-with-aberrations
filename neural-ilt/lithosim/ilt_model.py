import torch
import torch.utils.checkpoint as cp
from pytorch_lightning import LightningModule
import torchvision.models as models
from lt_simulator import LTSimulator
from transformers import get_linear_schedule_with_warmup

@torch.compile
class LithoZernikeRegressor(LightningModule):
    def __init__(self, model_name='efficientnet_b0', num_zernike=6, lr=1e-3, checkpointing: bool = False, warmup_steps=500):        
        super().__init__()
        self.save_hyperparameters()

        self.sim = LTSimulator(checkpointing=checkpointing)
        self.backbone = get_backbone(model_name, in_chans=2, num_classes=num_zernike)

        self.loss_func = torch.nn.functional.l1_loss
        self.checkpointing = checkpointing
        self.warmup_steps = warmup_steps



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
        loss = self.loss_func(modelled_imgs, aberr_imgs)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, z_true = batch
        litho_aberr_imgs = imgs[:, 1:] # skip design images
        z_preds = self(litho_aberr_imgs)
        design_imgs, litho_imgs, aberr_imgs = torch.split(imgs, 1, dim=1) # [B, 3, H, W] -> [B, (1, 1, 1), H, W]
        
        modelled_imgs = self.sim.run_lithosim(design_imgs, zernike_coeffs=z_preds)
        loss = self.loss_func(modelled_imgs, aberr_imgs)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]
    
    def on_after_backward(self):
        grads = []
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.detach().abs().mean().item()
                self.log(f'grad_mean_{n}', grad_norm, prog_bar=False, logger=True)
                grads.append(grad_norm)
        if grads:
            mean_grad = sum(grads) / len(grads)
            self.log('mean_grad', mean_grad, prog_bar=True, logger=True)


@torch.compile
class DualLithoZernikeRegressor(LithoZernikeRegressor):
    def __init__(self, model_name='efficientnet_b0', num_zernike=6, lr=1e-3, checkpointing: bool = False, warmup_steps=500, rev_weight: float = 10.):
        super().__init__(model_name, num_zernike, lr, checkpointing, warmup_steps)
        self.save_hyperparameters()

        self.sim = LTSimulator(checkpointing=checkpointing)
        self.backbone = get_backbone(model_name, in_chans=2, num_classes=num_zernike*2)

        self.loss_func = torch.nn.functional.l1_loss
        self.num_zernike = num_zernike
        self.checkpointing = checkpointing
        self.rev_weight = rev_weight
        


    def forward(self, x):
        if self.checkpointing:
            def run_func(input):
                return torch.split(self.backbone(input), self.num_zernike, dim=-1)
            return cp.checkpoint(run_func, x, use_reentrant=False)
        else:
            return torch.split(self.backbone(x), self.num_zernike, dim=-1)

    def training_step(self, batch, batch_idx):
        imgs, z_true = batch
        litho_aberr_imgs = imgs[:, 1:] # skip design images

        z_preds, z_rev_preds = self(litho_aberr_imgs)
        design_imgs, litho_imgs, aberr_imgs = torch.split(imgs, 1, dim=1) # [B, 3, H, W] -> [B, (1, 1, 1), H, W]
        
        modelled_imgs = self.sim.run_lithosim(design_imgs, zernike_coeffs=z_preds)
        modelling_loss = self.loss_func(modelled_imgs, aberr_imgs)

        reconstructed_imgs = self.sim.run_lithosim(aberr_imgs, zernike_coeffs=z_rev_preds)
        reconstruction_loss = self.loss_func(reconstructed_imgs, litho_imgs)
        zernike_loss = torch.nn.functional.mse_loss(z_preds, z_true)

        loss = modelling_loss + reconstruction_loss * self.rev_weight + zernike_loss
        self.log_dict({'modelling_loss': modelling_loss, 'reconstruction_loss': reconstruction_loss,
                 'zernike_loss': zernike_loss, 'lr': self.lr_schedulers().get_last_lr()[0]})
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, z_true = batch
        litho_aberr_imgs = imgs[:, 1:] # skip design images

        z_preds, z_rev_preds = self(litho_aberr_imgs)
        design_imgs, litho_imgs, aberr_imgs = torch.split(imgs, 1, dim=1) # [B, 3, H, W] -> [B, (1, 1, 1), H, W]
        
        modelled_imgs = self.sim.run_lithosim(design_imgs, zernike_coeffs=z_preds)
        modelling_loss = self.loss_func(modelled_imgs, aberr_imgs)

        reconstructed_imgs = self.sim.run_lithosim(aberr_imgs, zernike_coeffs=z_rev_preds)
        reconstruction_loss = self.loss_func(reconstructed_imgs, litho_imgs)
        zernike_loss = torch.nn.functional.mse_loss(z_preds, z_true)

        loss = modelling_loss + reconstruction_loss * self.rev_weight + zernike_loss
        self.log_dict({'val_modelling_loss': modelling_loss, 'val_reconstruction_loss': reconstruction_loss,
                 'val_zernike_loss': zernike_loss})
        self.log('val_loss', loss, prog_bar=True)
        return loss


def get_backbone(model_name: str, in_chans=2, pretrained=True, num_classes=20):
    backbone = getattr(models, model_name)(pretrained=pretrained)

    if model_name.startswith('resnet'):
        backbone.conv1 = torch.nn.Conv2d(
            in_chans, backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False)
        num_features = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_features, num_classes)
    elif model_name.startswith('efficientnet'):
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