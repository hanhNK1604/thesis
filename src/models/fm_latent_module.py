import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True)  

import torch
import lightning as L 
from torch import nn, optim 
import torchmetrics 
from torchmetrics.classification import BinaryJaccardIndex


from torch.optim import Optimizer
from torchvision.utils import make_grid

from flow_matching.solver import ODESolver 
from flow_matching.path import CondOTProbPath 
from flow_matching.utils.model_wrapper import ModelWrapper 

from src.models.flow_matching.flow_matching import VelocityModel 
from src.models.vae_module import VAEModule
from src.models.vae_mask_module import VAEMaskModule


class FlowLatent(ModelWrapper): 
    def __init__(self, net: nn.Module, w: float=4.0): 
        super(FlowLatent, self).__init__(model=net) 
        self.w = w 

    def forward(self, x, t, **extras):
        c = extras.get("c") 
        return (1 - self.w) * self.model.forward(x=x, t=t, c=None) + self.w * self.model.forward(x=x, t=t, c=c) 
            
class FlowMatchingLatentModule(L.LightningModule): 
    def __init__(self, 
        num_steps:int, 
        velocity_model: VelocityModel, 
        optimizer: Optimizer, 
        w: float = 4.0, 
        vae_mask_path: str = "dir",
        vae_image_path: str = "dir" 
    ): 
        super(FlowMatchingLatentModule, self).__init__()  
        self.save_hyperparameters(logger=False)
        self.num_steps = num_steps
        self.velocity_model = velocity_model
        self.optimizer = optimizer 
        self.w = w 

        self.mse_loss = nn.MSELoss() 

        self.model_wrapper = FlowLatent(net=self.velocity_model, w=self.w) 
        self.solver = ODESolver(velocity_model=self.model_wrapper)
        self.path = CondOTProbPath()

        self.iou = BinaryJaccardIndex()
        self.mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)

        self.vae_image_module = VAEModule.load_from_checkpoint(vae_image_path)
        self.vae_image_module.eval().freeze() 
        self.vae_image_model = self.vae_image_module.vae_model

        self.vae_mask_module = VAEMaskModule.load_from_checkpoint(vae_mask_path)
        self.vae_mask_module.eval().freeze() 
        self.vae_mask_model = self.vae_mask_module.vae_model

        
    def forward(self, x, t, c=None): 
        velocity = self.velocity_model.forward(x=x, t=t, c=c) 
        return velocity 

    def sample(self, input_size: torch.Size, z_image):

        noise = torch.randn(size=input_size, device="cuda") 
        extras = {"c": z_image} 
        z_mask_predict = self.solver.sample(x_init=noise, step_size=1.0/self.num_steps, method="midpoint", **extras) 
    
        mask = self.vae_mask_model.decode(z_mask_predict)
        mask = torch.sigmoid(mask) 
        mask_predict = (mask > 0.5).long() 

        return mask_predict 

    def rescale(self, image):
        return image * self.std.to(image.device) + self.mean.to(image.device) 
        
    def step(self, batch): 
        image, mask = batch 

        z_image, _ = self.vae_image_model.encode(image)
        z_mask, _ = self.vae_mask_model.encode(mask) 

        t = torch.rand(size=(z_mask.shape[0],), device=z_mask.device) 
        noise = torch.randn_like(z_mask, device=z_mask.device) 

        path_sample = self.path.sample(x_0=noise, x_1=z_mask, t=t) 
        z_t = path_sample.x_t.to(z_mask.device) 
        dz_t = path_sample.dx_t.to(z_mask.device) 

        velocity = None 
        rand_value = torch.rand(size=(1,)) 
        if rand_value[0] > 0.5: 
            velocity = self.forward(x=z_t, t=t, c=z_image) 
        else: 
            velocity = self.forward(x=z_t, t=t, c=None) 

        loss = self.mse_loss(velocity, dz_t)
        return loss

    def training_step(self, batch, batch_index): 
        loss = self.step(batch=batch) 
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True) 
        return loss

    def validation_step(self, batch, batch_index): 
        loss = self.step(batch=batch) 
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True) 

        image, mask = batch 

        z_image, _ = self.vae_image_model.encode(image) 
        z_mask, _ = self.vae_mask_model.encode(mask) 

        mask_pred = self.sample(input_size=z_mask.shape, z_image=z_image) 
        mask = (mask > 0.5).long()
        iou = self.iou(mask_pred, mask.long()) 
        self.log('val/iou', iou, prog_bar=True, on_step=False, on_epoch=True)

        if batch_index == 4: 
            labels = make_grid(mask.float(), nrow=2) 
            pred = make_grid(mask_pred.float(), nrow=2) 
            image = self.rescale(image) 
            image = make_grid(image, nrow=2)
            self.logger.log_image(images=[image], key='val/image')
            self.logger.log_image(images=[labels], key='val/mask')
            self.logger.log_image(images=[pred], key='val/mask_pred') 
    
    def test_step(self, batch, batch_index): 
        loss = self.step(batch=batch) 
        self.log('test/loss', loss, prog_bar=True, on_step=False, on_epoch=True) 

        image, mask = batch 

        z_image, _ = self.vae_image_model.encode(image) 
        z_mask, _ = self.vae_mask_model.encode(mask) 

        mask_pred = self.sample(input_size=z_mask.shape, z_image=z_image) 
        mask = (mask > 0.5).long()
        iou = self.iou(mask_pred, mask.long()) 
        self.log('test/iou', iou, prog_bar=True, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return self.optimizer(self.parameters()) 
