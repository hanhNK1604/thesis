import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True) 

import torch 
from torch import nn 
import pytorch_lightning as L
from torchvision.utils import make_grid 

from src.models.diffusion.DDIM import DDIMSampler 
from src.models.diffusion.LatentDiffusion import LatentDiffusion 

from torchmetrics.classification import BinaryJaccardIndex  

class LatentDiffusionModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: LatentDiffusion, 
        optimizer, 
        sampler: DDIMSampler, 
    ): 
        super(LatentDiffusionModule, self).__init__() 

        self.save_hyperparameters(logger=False)

        self.diffusion_model = diffusion_model 
        self.optimizer = optimizer 
        self.sampler = sampler 
        self.iou = BinaryJaccardIndex()
        self.loss_fn = nn.MSELoss() 
        

    def forward(self, batch): 
        pred_noise, noise = self.diffusion_model.forward(batch) 
        return pred_noise, noise 

    def training_step(self, batch, batch_index):
        pred_noise, noise = self.forward(batch) 
        loss = self.loss_fn(pred_noise, noise)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_index): 
        image, mask = batch 
        pred_noise, noise = self.forward(batch) 
        loss = self.loss_fn(pred_noise, noise)
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        self.sampler.denoise_net = self.diffusion_model.denoise_net  
        z_image = self.diffusion_model.encode_image(image)

        z_fake = self.sampler.reverse_process_condition(c=z_image, batch_size=z_image.shape[0]) 

        mask_pred = self.diffusion_model.decode_mask(z_fake) 

        iou = self.iou(mask_pred, (mask > 0.5).long())
        self.log('val/iou', iou, prog_bar=True, on_step=False, on_epoch=True) 

        if batch_index == 16: 
            image = make_grid(self.diffusion_model.rescale(image), nrow=2) 
            mask = make_grid((mask > 0.5).long(), nrow=2) 
            mask_pred = make_grid(mask_pred, nrow=2) 

            self.logger.log_image(images=[image], key="val/image") 
            self.logger.log_image(images=[mask], key="val/mask") 
            self.logger.log_image(images=[mask_pred], key="val/mask_pred")
    
    def test_step(self, batch, batch_index): 
        image, mask = batch 
        pred_noise, noise = self.forward(batch) 
        loss = self.loss_fn(pred_noise, noise)
        self.log('test/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        self.sampler.denoise_net = self.diffusion_model.denoise_net  
        z_image = self.diffusion_model.encode_image(image)

        z_fake = self.sampler.reverse_process_condition(c=z_image, batch_size=z_image.shape[0]) 

        mask_pred = self.diffusion_model.decode_mask(z_fake) 

        iou = self.iou(mask_pred, (mask > 0.5).long())
        self.log('test/iou', iou, prog_bar=True, on_step=False, on_epoch=True) 

        if batch_index == 16: 
            image = make_grid(self.diffusion_model.rescale(image), nrow=2) 
            mask = make_grid((mask > 0.5).long(), nrow=2) 
            mask_pred = make_grid(mask_pred, nrow=2) 

            self.logger.log_image(images=[image], key="test/image") 
            self.logger.log_image(images=[mask], key="test/mask") 
            self.logger.log_image(images=[mask_pred], key="test/mask_pred")

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
    