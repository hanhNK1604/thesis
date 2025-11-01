import lightning as L 
import torch 
from torch import nn 
from torchvision.utils import make_grid 
import torch.nn.functional as F

import torchmetrics

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 
from src.models.vae.net.kl_vae import KLVAEModel
from src.models.vae.components.FeatureExtractor import FeatureExtractor 
from torchmetrics.classification import BinaryJaccardIndex

class VAEMaskModule(L.LightningModule): 
    def __init__(
        self, 
        vae_model: KLVAEModel,
        optimizer
    ): 
        super(VAEMaskModule, self).__init__()

        self.save_hyperparameters(logger=False)
        self.vae_model = vae_model 
        self.optimizer = optimizer 
        
        self.iou = BinaryJaccardIndex()

        self.clf_loss = nn.BCEWithLogitsLoss()
    
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0) 
        self.ssim_metric = StructuralSimilarityIndexMeasure()

    def dice_loss(self, res_image, image): 
        bs = res_image.shape[0]
        res_image = torch.sigmoid(res_image) 
        res_image_flat = res_image.view(bs, -1) 
        image_flat = image.view(bs, -1) 

        intersection = (res_image_flat * image_flat).sum(dim=1) 
        dice = (2.0 * intersection + 1e-6) / (res_image_flat.sum(dim=1) + image_flat.sum(dim=1) + 1e-6) 
        dice = 1.0 - dice 
        return dice.mean()       

    def bce_loss(self, res_image, image): 
        return F.binary_cross_entropy_with_logits(res_image, image)


    def forward(self, x): 
        res_image, losses = self.vae_model.forward(x) #losses only contain kld_loss 
        dice_loss = self.dice_loss(res_image=res_image, image=x)  
        bce_loss = self.bce_loss(res_image=res_image, image=x) 
        losses['dice_loss'] = 0.7 * dice_loss
        losses['bce_loss'] = 0.3 * bce_loss

        return res_image, losses

    
    def training_step(self, batch, batch_index): 
        res_image, losses = self.forward(batch) 
        
        total_loss = sum(losses.values())
        self.log('train/total_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False) 

        for key in losses.keys(): 
            self.log(f'train/{key}', losses[key].detach(), on_step=False, on_epoch=True) 
        
        return total_loss

    def interpolation(self, batch): 
        
        latens, _ = self.vae_model.encode(batch) 
        steps = torch.linspace(start=0, end=1, steps=25) 

        start_latent = latens[0].unsqueeze(0)
        end_latent = latens[1].unsqueeze(0)
        minus_latent = end_latent - start_latent 

        list_latent_interpolation = [start_latent + minus_latent * i for i in steps] 
        list_decode_interpolation = [torch.sigmoid(self.vae_model.decode(latent)) for latent in list_latent_interpolation] 
        list_decode_interpolation = torch.cat(list_decode_interpolation, dim=0) 

        image = make_grid(list_decode_interpolation, nrow=5)

        return image 


    def validation_step(self, batch, batch_index): 
        res_image, losses = self.forward(batch) 
        res_image = torch.sigmoid(res_image) 

        ssim = self.ssim_metric(res_image, batch) 
        psnr = self.psnr_metric(res_image, batch) 
        
        
        iou = self.iou((res_image > 0.5).long(), (batch > 0.5).long()) 

        self.log('val/iou', iou, on_epoch=True, on_step=False) 
        self.log('val/ssim', ssim, on_epoch=True, on_step=False) 
        self.log('val/psnr', psnr, on_epoch=True, on_step=False) 
        
        total_loss = sum(losses.values())
        self.log('val/total_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False) 
        
        for key in losses.keys(): 
            self.log(f'val/{key}', losses[key].detach(), on_step=False, on_epoch=True) 

        if batch_index == 16:  
            # random_image = self.sample() 
            # self.logger.log_image(images=[random_image], key='val/random_image')

            fake_image = make_grid(res_image, nrow=2) 
            real_image = make_grid(batch, nrow=2) 

            self.logger.log_image(images=[real_image], key='val/real_image')
            self.logger.log_image(images=[fake_image], key='val/fake_image') 

            image_interpolation = self.interpolation(batch) 

            self.logger.log_image(images=[image_interpolation], key='val/interpolation')
        
    def test_step(self, batch, batch_index): 
        # res_image, losses = self.forward(batch) 

        # ssim = self.ssim_metric(self.rescale(res_image), self.rescale(batch)) 
        # psnr = self.psnr_metric(self.rescale(res_image), self.rescale(batch)) 

        # mask = (self.rescale(batch) > 0.5).long() 
        # mask_pred = (self.rescale(res_image) > 0.5).long() 
        # iou = self.iou(mask_pred, mask) 

        res_image, losses = self.forward(batch) 
        res_image = torch.sigmoid(res_image) 

        ssim = self.ssim_metric(res_image, batch) 
        psnr = self.psnr_metric(res_image, batch) 
        
        iou = self.iou((res_image > 0.5).long(), (batch > 0.5).long()) 

        self.log('test/iou', iou, on_epoch=True, on_step=False)
        self.log('test/ssim', ssim, on_epoch=True, on_step=False) 
        self.log('test/psnr', psnr, on_epoch=True, on_step=False) 
            
    def configure_optimizers(self): 
        return self.optimizer(self.parameters())

# a = torch.randn(1, 2, 2, device='cuda')
# print(a)