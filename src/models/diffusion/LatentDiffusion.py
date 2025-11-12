import torch 
from torch import nn 
import math

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)  

from src.models.diffusion.UNet import UNet 
from src.models.vae_mask_module import VAEMaskModule 
from src.models.vae_module import VAEModule 

class LatentDiffusion(nn.Module): 
    def __init__(
        self, 
        denoise_net: UNet, 
        time_steps: int, 
        schedule: str,
        vae_image_path: str, 
        vae_mask_path: str
    ): 
        super(LatentDiffusion, self).__init__() 
        self.denoise_net = denoise_net
        self.time_steps = time_steps 
        self.schedule = schedule 
        self.vae_image_module = VAEModule.load_from_checkpoint(vae_image_path) 
        self.vae_mask_module = VAEMaskModule.load_from_checkpoint(vae_mask_path) 
        self.vae_image_module.eval().freeze() 
        self.vae_mask_module.eval().freeze() 

        if schedule == 'linear': 
            self.betas = torch.linspace(start=0.0015, end=0.019, steps=time_steps) 
        elif schedule == 'cosine': 
            s = 8e-3
            t = torch.arange(time_steps + 1, dtype=torch.float) / time_steps + s
            alpha_bar = torch.cos(t / (1 + s) * math.pi / 2).pow(2)
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = betas.clamp(max=0.999)
        
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar) 

        self.mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)

    def forward_process(self, x, noise, t): 
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(t.device) 
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(t.device)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None, None].to(x.device) 
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None].to(x.device) 
        noise = noise.to(x.device) 

        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise 
    
    def encode_image(self, image): 
        z_image, _ = self.vae_image_module.vae_model.encode(image) 
        return z_image 
        # return torch.randn(size=(4, 3, 64, 64))

    def encode_mask(self, mask): 
        z_mask, _ = self.vae_mask_module.vae_model.encode(mask) 
        return z_mask 
        # return torch.randn(size=(4, 3, 64, 64))
    
    def decode_mask(self, z_mask): 
        mask = self.vae_mask_module.vae_model.decode(z_mask) 
        return (torch.sigmoid(mask) > 0.5).long() 
        # pass 
    
    def decode_image(self, z_image): 
        return self.vae_image_module.vae_model.decode(z_image) 
    
    def rescale(self, image): 
        return image * self.std.to(image.device) + self.mean.to(image.device) 

    
    def forward(self, batch): 
        image, mask = batch 
        z_image = self.encode_image(image) 
        z_mask = self.encode_mask(mask) 

        self.denoise_net = self.denoise_net.to(z_image.device) 
        t = torch.randint(low=0, high=self.time_steps, size=(z_mask.shape[0],), device=z_mask.device)

        noise = torch.randn_like(z_mask, device=z_mask.device) 
        zt_mask = self.forward_process(z_mask, noise, t) 

        cond = torch.randint(low=0, high=2, size=(1,)) # 0 is no condition, else yes 
        if cond[0] == 0:
            pred_noise = self.denoise_net.forward(zt_mask, t, c=None) 
        else: 
            pred_noise = self.denoise_net.forward(zt_mask, t, c=z_image) 
        return pred_noise, noise 
    

# unet = UNet(in_ch=3, t_emb_dim=256, base_channel=64, multiplier=[1, 2, 2, 4], use_attention=True, ratio=1)
# latent_diff = LatentDiffusion(denoise_net=unet, time_steps=1000, schedule="cosine", vae_image_path=None, vae_mask_path=None) 

# image = torch.randn(size=(4, 3, 256, 256)) 
# mask = torch.randn(size=(4, 3, 256, 256)) 

# batch = image, mask 
# print(latent_diff.forward(batch)[0].shape)
# print(latent_diff.forward(batch)[1].shape)

# print(latent_diff.forward(batch)[0])
