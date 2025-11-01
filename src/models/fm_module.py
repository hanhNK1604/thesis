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


class Flow(ModelWrapper): 
    def __init__(self, net: nn.Module, w: float=4.0): 
        super(Flow, self).__init__(model=net) 
        self.w = w 

    def forward(self, x, t, **extras):
        c = extras.get("c") 
        return (1 - self.w) * self.model.forward(x=x, t=t, c=None) + self.w * self.model.forward(x=x, t=t, c=c) 
            
class FlowMatchingModule(L.LightningModule): 
    def __init__(self, 
        num_steps:int, 
        velocity_model: VelocityModel, 
        optimizer: Optimizer, 
        w: float = 4.0, 
        vae_path: str = "dir" 
    ): 
        super(FlowMatchingModule, self).__init__()  
        self.save_hyperparameters(logger=False)
        self.num_steps = num_steps
        self.velocity_model = velocity_model
        self.optimizer = optimizer 
        self.w = w 

        self.mse_loss = nn.MSELoss() 

        self.model_wrapper = Flow(net=self.velocity_model, w=self.w) 
        self.solver = ODESolver(velocity_model=self.model_wrapper)
        self.path = CondOTProbPath()

        self.iou = BinaryJaccardIndex()
        self.mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)

        self.vae_module = VAEModule.load_from_checkpoint(vae_path)
        self.vae_module.eval().freeze() 
        self.vae_model = self.vae_module.vae_model

        

    # def dice_loss(self, pred, target): 

    #     pred = pred.contiguous()
    #     target = target.contiguous()

    #     intersection = (pred * target).sum(dim=(2, 3))
    #     dice = (2. * intersection + 1e-6) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6)
    #     loss = 1 - dice.mean()
    #     return loss

        
    def forward(self, x, t, c=None): 
        velocity = self.velocity_model.forward(x=x, t=t, c=c) 
        return velocity 

    def sample(self, input_size: torch.Size, c): 
        noise = torch.randn(size=input_size, device="cuda") 
        extras = {"c": c} 
        mask_predict = self.solver.sample(x_init=noise, step_size=1.0/self.num_steps, method="midpoint", **extras) 
        mask_predict = torch.clamp(mask_predict, min=0, max=1) 
        mask_predict = (mask_predict > 0.5)

        return mask_predict 

    def rescale(self, image):
        return image * self.std.to(image.device) + self.mean.to(image.device) 
        

    def step(self, batch): 
        image, mask = batch 
        image_c, _ = self.vae_model.encode(image)
        t = torch.rand(size=(mask.shape[0],), device=mask.device) 
        noise = torch.randn_like(mask, device=mask.device) 

        path_sample = self.path.sample(x_0=noise, x_1=mask, t=t)
        x_t = path_sample.x_t.to(mask.device) 
        dx_t = path_sample.dx_t.to(mask.device)

        velocity = None
       
        rand_value = torch.rand(size=(1,)) 
        if rand_value[0] > 0.5: 
            velocity = self.forward(x=x_t, t=t, c=image_c) 
        else: 
            velocity = self.forward(x=x_t, t=t, c=None) 


        loss = self.mse_loss(velocity, dx_t)
        return loss 

    def training_step(self, batch, batch_index): 
        loss = self.step(batch=batch) 
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True) 
        return loss

    def validation_step(self, batch, batch_index): 
        loss = self.step(batch=batch) 
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True) 

        image, mask = batch 
        image_c, _ = self.vae_model.encode(image)
        mask_pred = self.sample(input_size=mask.shape, c=image_c) 

        mask = (mask > 0.5).long()

        iou = self.iou(mask_pred.long(), mask.long()) 
        self.log('val/iou', iou, prog_bar=True, on_step=False, on_epoch=True)

        if batch_index == 10: 
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
        image_c, _ = self.vae_model.encode(image)
        mask_pred = self.sample(input_size=mask.shape, c=image_c) 

        mask = (mask > 0.5).long()

        iou = self.iou(mask_pred.long(), mask.long()) 
        self.log('test/iou', iou, prog_bar=True, on_step=False, on_epoch=True)

        if batch_index == 10: 
            labels = make_grid(mask.float(), nrow=2) 
            pred = make_grid(mask_pred.float(), nrow=2) 
            image = self.rescale(image) 
            image = make_grid(image, nrow=2)
            self.logger.log_image(images=[image], key='test/image')
            self.logger.log_image(images=[labels], key='test/mask')
            self.logger.log_image(images=[pred], key='test/mask_pred') 


    def configure_optimizers(self):
        return self.optimizer(self.parameters()) 
