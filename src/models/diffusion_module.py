# import rootutils 
# rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True)  

# import torch
# import lightning as L 
# from torch import nn, optim 
# from torch.optim import Optimizer
# import torchmetrics 
# from torchmetrics.classification import BinaryJaccardIndex
# from torch.optim import Optimizer
# from torchvision.utils import make_grid

# from src.models.diffusion.DiffusionModel import DiffusionModel 


# class DiffusionModule(L.LightningModule): 
#     def __init__(
#         self, 
#         diffusion_model: DiffusionModel, 
#         optimizer 
#     ): 
#         super(DiffusionModule, self).__init__() 
#         self.save_hyperparameters(logger=False) 
#         self.diffusion_model = diffusion_model 
#         self.optimizer = optimizer 

#         self.loss_fn = nn.MSELoss() 
#         self.iou = BinaryJaccardIndex() 


#     def forward(self, batch): 
#         image_resize, mask_resize, image_original, mask_original = batch 
#         pred_noise, noise = self.diffusion_model.forward(image=image_resize, mask=mask_resize) 
#         return pred_noise, noise 

#     def training_step(self, batch, batch_index): 
#         pred_noise, noise = self.forward(batch=batch) 
#         loss = self.loss_fn(pred_noise, noise) 
#         self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False) 

#         return loss 
    
#     def validation_step(self, batch, batch_index): 
#         image_resize, mask_resize, image_original, mask_original = batch  
#         pred_noise, noise = self.diffusion_model.forward(image=image_resize, mask=mask_resize) 
#         loss = self.loss_fn(pred_noise, noise) 
#         self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

#         mask_pred = self.diffusion_model.sample(image=image_resize)
#         iou = self.iou(mask_pred, (mask_original > 0.5).long())
#         self.log('val/iou', iou, on_epoch=True, on_step=False, prog_bar=True) 

#         mask_pred_n = self.sample_n(image=image_resize, n=5) 
#         iou_n = self.iou(mask_pred_n, (mask_original > 0.5).long()) 
#         self.log('val/iou_n', iou_n, on_epoch=True, on_step=False, prog_bar=True) 

#         if batch_index == 16: 
#             mask_pred_n = make_grid(mask_pred_n, nrow=2) 
#             mask_pred = make_grid(mask_pred, nrow=2) 
#             mask_original = make_grid((mask_original > 0.5).long(), nrow=2) 
#             image_original = make_grid(image_original, nrow=2)
            
#             self.logger.log_image(images=[mask_pred_n], key='val/mask_pred_n')
#             self.logger.log_image(images=[mask_pred], key='val/mask_pred')
#             self.logger.log_image(images=[mask_original], key='val/mask')
#             self.logger.log_image(images=[image_original], key='val/image')

#     def sample_n(self, image: torch.Tensor, n: int = 5): 
#         list_mask = [] 
#         for i in range(n): 
#             mask_pred = self.diffusion_model.sample(image=image) 
#             list_mask.append(torch.unsqueeze(mask_pred, dim=0)) 
#         res = torch.cat(list_mask, dim=0) 
#         res = torch.mean(res.float(), dim=0) 
#         return (res > 0.5).long()  



#     def test_step(self, batch, batch_index): 
#         image_resize, mask_resize, image_original, mask_original = batch  
#         pred_noise, noise = self.diffusion_model.forward(image=image_resize, mask=mask_resize) 
#         loss = self.loss_fn(pred_noise, noise) 
#         self.log('test/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

#         mask_pred = self.diffusion_model.sample(image=image_resize)
#         iou = self.iou(mask_pred, (mask_original > 0.5).long())
#         self.log('test/iou', iou, on_epoch=True, on_step=False, prog_bar=True) 

#         mask_pred_n = self.sample_n(image=image_resize, n=5) 
#         iou_n = self.iou(mask_pred_n, (mask_original > 0.5).long()) 
#         self.log('test/iou_n', iou_n, on_epoch=True, on_step=False, prog_bar=True) 

#         if batch_index == 16: 
#             mask_pred_n = make_grid(mask_pred_n, nrow=2) 
#             mask_pred = make_grid(mask_pred, nrow=2) 
#             mask_original = make_grid((mask_original > 0.5).long(), nrow=2) 
#             image_original = make_grid(image_original, nrow=2)
            
#             self.logger.log_image(images=[mask_pred_n], key='test/mask_pred_n')
#             self.logger.log_image(images=[mask_pred], key='test/mask_pred')
#             self.logger.log_image(images=[mask_original], key='test/mask')
#             self.logger.log_image(images=[image_original], key='test/image')
        
            
#     def configure_optimizers(self):
#         return self.optimizer(self.parameters()) 


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


class DiffusionModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: DiffusionModel, 
        optimizer,
        cfg_scale: float = 4.0   
    ): 
        super(DiffusionModule, self).__init__() 
        self.save_hyperparameters(logger=False) 
        self.diffusion_model = diffusion_model 
        self.optimizer = optimizer 
        self.cfg_scale = cfg_scale

        self.loss_fn = nn.MSELoss() 
        self.iou = BinaryJaccardIndex() 


    def forward(self, batch): 
        image_resize, mask_resize, image_original, mask_original = batch 
        mask_resize_norm = (mask_resize * 2.0) - 1.0
        image_resize_norm = (image_resize * 2.0) - 1.0
        
        pred_noise, noise = self.diffusion_model.forward(image=image_resize_norm, mask=mask_resize_norm) 
        return pred_noise, noise 

    def training_step(self, batch, batch_index): 
        pred_noise, noise = self.forward(batch=batch) 
        loss = self.loss_fn(pred_noise, noise) 
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False) 

        return loss 
    
    def validation_step(self, batch, batch_index): 
        image_resize, mask_resize, image_original, mask_original = batch  
        
        mask_resize_norm = (mask_resize * 2.0) - 1.0
        image_resize_norm = (image_resize * 2.0) - 1.0
        
        pred_noise, noise = self.diffusion_model.forward(image=image_resize_norm, mask=mask_resize_norm) 
        loss = self.loss_fn(pred_noise, noise) 
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        mask_pred = self.diffusion_model.sample(image=image_resize_norm, cfg_scale=self.cfg_scale)
        
        iou = self.iou(mask_pred, (mask_original > 0.5).long())
        self.log('val/iou', iou, on_epoch=True, on_step=False, prog_bar=True) 

        mask_pred_n = self.sample_n(image=image_resize_norm, n=5) 
        iou_n = self.iou(mask_pred_n, (mask_original > 0.5).long()) 
        self.log('val/iou_n', iou_n, on_epoch=True, on_step=False, prog_bar=True) 

        if batch_index == 16: 
            mask_pred_n = make_grid(mask_pred_n, nrow=2) 
            mask_pred = make_grid(mask_pred, nrow=2) 
            mask_original = make_grid((mask_original > 0.5).long(), nrow=2) 
            image_original = make_grid(image_original, nrow=2)
            
            self.logger.log_image(images=[mask_pred_n], key='val/mask_pred_n')
            self.logger.log_image(images=[mask_pred], key='val/mask_pred')
            self.logger.log_image(images=[mask_original], key='val/mask')
            self.logger.log_image(images=[image_original], key='val/image')

    def sample_n(self, image: torch.Tensor, n: int = 5): 
        list_mask = [] 
        for i in range(n): 
            mask_pred = self.diffusion_model.sample(image=image, cfg_scale=self.cfg_scale) 
            list_mask.append(torch.unsqueeze(mask_pred, dim=0)) 
        res = torch.cat(list_mask, dim=0) 
        res = torch.mean(res.float(), dim=0) 
        return (res > 0.5).long()  


    def test_step(self, batch, batch_index): 
        image_resize, mask_resize, image_original, mask_original = batch  
        
        mask_resize_norm = (mask_resize * 2.0) - 1.0
        image_resize_norm = (image_resize * 2.0) - 1.0
        
        pred_noise, noise = self.diffusion_model.forward(image=image_resize_norm, mask=mask_resize_norm) 
        loss = self.loss_fn(pred_noise, noise) 
        self.log('test/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        mask_pred = self.diffusion_model.sample(image=image_resize_norm, cfg_scale=self.cfg_scale)
        iou = self.iou(mask_pred, (mask_original > 0.5).long())
        self.log('test/iou', iou, on_epoch=True, on_step=False, prog_bar=True) 

        mask_pred_n = self.sample_n(image=image_resize_norm, n=5) 
        iou_n = self.iou(mask_pred_n, (mask_original > 0.5).long()) 
        self.log('test/iou_n', iou_n, on_epoch=True, on_step=False, prog_bar=True) 

        if batch_index == 16: 
            mask_pred_n = make_grid(mask_pred_n, nrow=2) 
            mask_pred = make_grid(mask_pred, nrow=2) 
            mask_original = make_grid((mask_original > 0.5).long(), nrow=2) 
            image_original = make_grid(image_original, nrow=2)
            
            self.logger.log_image(images=[mask_pred_n], key='test/mask_pred_n')
            self.logger.log_image(images=[mask_pred], key='test/mask_pred')
            self.logger.log_image(images=[mask_original], key='test/mask')
            self.logger.log_image(images=[image_original], key='test/image')
        
            
    def configure_optimizers(self):
        return self.optimizer(self.parameters())