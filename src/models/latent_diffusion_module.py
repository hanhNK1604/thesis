import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True)  

import torch
import lightning as L 
from torch import nn, optim 
from torch.optim import Optimizer
import torchmetrics 
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.utils import make_grid

from src.models.diffusion.DiffusionModel import LatentDiffusionModel 

class LatentDiffusionModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: LatentDiffusionModel, 
        optimizer,
        cfg_scale: float = 4.0
    ): 
        super(LatentDiffusionModule, self).__init__() 
        self.save_hyperparameters(logger=False) 
        self.diffusion_model = diffusion_model 
        self.optimizer = optimizer 
        self.cfg_scale = cfg_scale 

        self.loss_fn = nn.MSELoss() 
        self.iou = BinaryJaccardIndex() 

        self.mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)

    def rescale(self, image):
        return image * self.std.to(image.device) + self.mean.to(image.device)

    def forward(self, batch): 
        _, _, image, mask = batch 

        z_image = self.diffusion_model.encode_image(image=image) 
        z_mask = self.diffusion_model.encode_mask(mask=mask) 

        noise_pred, noise = self.diffusion_model.forward(z_image=z_image, z_mask=z_mask) 
        return noise_pred, noise 
    
    def training_step(self, batch, batch_index): 
        noise_pred, noise = self.forward(batch=batch) 
        loss = self.loss_fn(noise_pred, noise) 
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True) 

        return loss 

    def validation_step(self, batch, batch_index): 
        _, _, image, mask = batch 

        z_image = self.diffusion_model.encode_image(image=image) 
        z_mask = self.diffusion_model.encode_mask(mask=mask) 

        noise_pred, noise = self.diffusion_model.forward(z_image=z_image, z_mask=z_mask) 
        loss = self.loss_fn(noise_pred, noise) 
        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True) 

        if batch_index == 16: 
            mask_pred = self.diffusion_model.sample(z_image=z_image, cfg_scale=self.cfg_scale) 
            mask_pred_n = self.sample_n(z_image=z_image, n=5)
            mask_vae = self.diffusion_model.vae_mask_module.vae_model.decode(z_mask) 
            mask_vae = (torch.sigmoid(mask_vae) > 0.5).long()
            image_vae = self.diffusion_model.vae_image_module.vae_model.decode(z_image) 
            image_vae = torch.clamp(self.rescale(image=image), 0, 1)
            image = self.rescale(image) 

            mask_pred = make_grid(mask_pred, nrow=2) 
            mask_pred_n = make_grid(mask_pred_n, nrow=2) 
            mask_vae = make_grid(mask_vae, nrow=2) 
            image_vae = make_grid(image_vae, nrow=2) 
            mask = make_grid((mask > 0.5).long(), nrow=2) 
            image = make_grid(image, nrow=2) 

            self.logger.log_image(images=[mask_pred], key='val/mask_pred')
            self.logger.log_image(images=[mask_pred_n], key='val/mask_pred_n') 
            self.logger.log_image(images=[mask_vae], key='val/mask_vae') 
            self.logger.log_image(images=[image_vae], key='val/image_vae') 
            self.logger.log_image(images=[mask], key='val/mask') 
            self.logger.log_image(images=[image], key='val/image') 

    def test_step(self, batch, batch_index): 
        _, _, image, mask = batch 
        z_image = self.diffusion_model.encode_image(image=image) 
        z_mask = self.diffusion_model.encode_mask(mask=mask) 

        noise_pred, noise = self.diffusion_model.forward(z_image=z_image, z_mask=z_mask) 
        loss = self.loss_fn(noise_pred, noise) 
        self.log('test/loss', loss, on_epoch=True, on_step=False, prog_bar=True) 

        mask_pred = self.diffusion_model.sample(z_image=z_image, cfg_scale=self.cfg_scale) 
        mask_pred_n = self.sample_n(z_image=z_image, n=5)

        iou = self.iou(mask_pred, (mask > 0.5).long()) 
        self.log('test/iou', iou, on_epoch=True, on_step=False, prog_bar=True) 
        iou_n = self.iou(mask_pred_n, (mask > 0.5).long()) 
        self.log('test/iou_n', iou_n, on_epoch=True, on_step=False, prog_bar=True)
        
        if batch_index == 16: 
            mask_vae = self.diffusion_model.vae_mask_module.vae_model.decode(z_mask) 
            mask_vae = (torch.sigmoid(mask_vae) > 0.5).long()
            image_vae = self.diffusion_model.vae_image_module.vae_model.decode(z_image) 
            image_vae = torch.clamp(self.rescale(image=image), 0, 1)
            image = self.rescale(image) 

            mask_pred = make_grid(mask_pred, nrow=2) 
            mask_pred_n = make_grid(mask_pred_n, nrow=2) 
            mask_vae = make_grid(mask_vae, nrow=2) 
            image_vae = make_grid(image_vae, nrow=2) 
            mask = make_grid((mask > 0.5).long(), nrow=2) 
            image = make_grid(image, nrow=2) 

            self.logger.log_image(images=[mask_pred], key='val/mask_pred')
            self.logger.log_image(images=[mask_pred_n], key='val/mask_pred_n') 
            self.logger.log_image(images=[mask_vae], key='val/mask_vae') 
            self.logger.log_image(images=[image_vae], key='val/image_vae') 
            self.logger.log_image(images=[mask], key='val/mask') 
            self.logger.log_image(images=[image], key='val/image') 


    def sample_n(self, z_image: torch.Tensor, n: int = 5): 
        samples = [] 
        for i in range(n): 
            sample = self.diffusion_model.sample(z_image=z_image, cfg_scale=self.cfg_scale)
            samples.append(torch.unsqueeze(sample, dim=0)) 
        
        res = torch.cat(samples, dim=0) 
        res = torch.mean(res.float(), dim=0) 
        res = (res > 0.5).long() 
        return res 
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters())
