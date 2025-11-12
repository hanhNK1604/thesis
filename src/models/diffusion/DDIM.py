import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

import torch 
from tqdm import tqdm 


from src.models.diffusion.LatentDiffusion import LatentDiffusion 
from src.models.diffusion.UNet import UNet 

class DDIMSampler: 
    def __init__(
        self, 
        diffusion_model: LatentDiffusion, 
        num_samples: int,
        image_size: int, 
        channels: int, 
        reduce_steps: int,  
        device: str, 
        w: int 
    ): 
        self.diffusion_model = diffusion_model.to(torch.device(device))
        self.num_samples = num_samples 
        self.image_size = image_size 
        self.channels = channels 
        self.reduce_steps = reduce_steps 
        self.device = torch.device(device) 
        self.w = torch.tensor([w], device=self.device)

        self.denoise_net = self.diffusion_model.denoise_net.to(self.device)

        self.time_steps = self.diffusion_model.time_steps 
        self.betas = self.diffusion_model.betas.to(self.device) 
        self.alphas = self.diffusion_model.alphas.to(self.device)
        self.alpha_bar = self.diffusion_model.alpha_bar.to(self.device) 
        self.sqrt_alpha_bar = self.diffusion_model.sqrt_alpha_bar.to(self.device) 
        self.sqrt_one_minus_alpha_bar = self.diffusion_model.sqrt_one_minus_alpha_bar.to(self.device)

        self.tau = [i for i in range(0, self.time_steps, self.time_steps//self.reduce_steps)]
        self.tau = [i for i in reversed(self.tau)] 


    def reverse_process_condition(self, c, batch_size=None): 
        c = c.to(self.device) 

        if batch_size is None: 
            batch_size = self.num_samples 

        self.denoise_net.eval()
        with torch.no_grad(): 
            x = torch.randn(size=(batch_size, self.channels, self.image_size, self.image_size), device=self.device) 
            for i in range(len(self.tau)): 
                if self.tau[i] != 0: 
                    t = (torch.ones(size=(batch_size,)) * self.tau[i]).long().to(self.device)
                    t_prev = (torch.ones(size=(batch_size,)) * self.tau[i + 1]).long().to(self.device)

                    pred_noise_no_cond = self.denoise_net.forward(x, t, c=None)     
                    pred_noise_with_cond = self.denoise_net(x, t, c=c)
                    final_noise_pred = (1 + self.w) * pred_noise_with_cond - self.w * pred_noise_no_cond 

                    alpha_bar = self.alpha_bar[t][:, None, None, None ]
                    alpha_bar_prev = self.alpha_bar[t_prev][:, None, None, None] 

                    pred_x0 = (x - torch.sqrt(1. - alpha_bar) * final_noise_pred)/torch.sqrt(alpha_bar)  

                    x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * final_noise_pred
                
                else: 
                    t = (torch.ones(size=(batch_size,)) * self.tau[i]).long().to(self.device)

                    pred_noise_no_cond = self.denoise_net(x, t, c=None)  
                    pred_noise_with_cond = self.denoise_net(x, t, c=c)
                    final_noise_pred = (1 + self.w) * pred_noise_with_cond - self.w * pred_noise_no_cond 

                    alpha_bar = self.alpha_bar[t][:, None, None, None ]

                    x = (x - torch.sqrt(1. - alpha_bar) * final_noise_pred)/torch.sqrt(alpha_bar)
                
            return x
        
# unet = UNet(in_ch=3, t_emb_dim=256, base_channel=64, multiplier=[1, 2, 2, 4], use_attention=True, ratio=1)
# latent_diff = LatentDiffusion(denoise_net=unet, time_steps=1000, schedule="cosine", vae_image_path=None, vae_mask_path=None) 
# sampler = DDIMSampler(diffusion_model=latent_diff, num_samples=4, image_size=64, channels=3, reduce_steps=5, device="cpu", w=4.0) 


# image = torch.randn(size=(4, 3, 256, 256)) 
# mask = torch.randn(size=(4, 3, 256, 256)) 

# batch = image, mask 
# print(latent_diff.forward(batch)[0].shape)
# print(latent_diff.forward(batch)[1].shape)

# z_image = latent_diff.encode_image(image) 
# x = sampler.reverse_process_condition(c=z_image, batch_size=z_image.shape[0])
# print(x.shape)
