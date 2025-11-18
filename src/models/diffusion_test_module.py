
import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True)  

import torch
import lightning as L 
from torch import nn, optim 
from torch.optim import Optimizer
import torchmetrics 
from torchmetrics.classification import BinaryJaccardIndex
from torch.optim import Optimizer
from torchvision.utils import make_grid

from src.models.diffusion.DiffusionModel import DiffusionModel 


class DiffusionTestModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: DiffusionModel, 
        optimizer, 
    ): 
        super(DiffusionTestModule, self).__init__() 
        self.save_hyperparameters(logger=False) 
        self.diffusion_model = diffusion_model 
        self.optimizer = optimizer 


        self.loss_fn = nn.MSELoss() 


    def forward(self, batch): 
        image, label = batch 
        noise_pred, noise = self.diffusion_model.forward(image=image) 
        return noise_pred, noise 
       

    def training_step(self, batch, batch_index): 
        noise_pred, noise = self.forward(batch=batch) 
        loss = self.loss_fn(noise_pred, noise) 
        self.log('train/loss', loss, prog_bar=True, on_epoch=True, on_step=False) 
        return loss 
    
    def validation_step(self, batch, batch_index): 
        noise_pred, noise = self.forward(batch=batch) 
        loss = self.loss_fn(noise_pred, noise) 
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, on_step=False) 
        return loss

    def on_validation_epoch_end(self):
        image = self.diffusion_model.sample()
        image = make_grid(image, nrow=5) 
        self.logger.log_image(images=[image], key="val/image_gen")


    def test_step(self, batch, batch_index): 
        noise_pred, noise = self.forward(batch=batch) 
        loss = self.loss_fn(noise_pred, noise) 
        self.log('test/loss', loss, prog_bar=True, on_epoch=True, on_step=False) 
        return loss
        
            
    def configure_optimizers(self):
        return self.optimizer(self.parameters())