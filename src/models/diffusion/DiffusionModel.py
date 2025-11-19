import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch 
from torch import nn 
from torchvision import transforms
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from tqdm import tqdm 

from src.models.vae_module import VAEModule 
from src.models.vae_mask_module import VAEMaskModule 

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(inplace=True), 
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch)
        )

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1) 
        else: 
            self.res_conv = nn.Identity() 
        
    def forward(self, x):
        
        out = self.conv(x) 
        res = self.res_conv(x) 

        return out + res 

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim=256, scale_factor=2):
        super(DownBlock, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_ch=in_ch, out_ch=out_ch),
            ResBlock(in_ch=out_ch, out_ch=out_ch),
            ResBlock(in_ch=out_ch, out_ch=out_ch)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch),
            nn.Linear(in_features=out_ch, out_features=out_ch)
        )

        self.c_emb_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear"), 
            ResBlock(in_ch=3, out_ch=out_ch),
            ResBlock(in_ch=out_ch, out_ch=out_ch),
        )


    def forward(self, x, t, c=None):
        """
        params:
          x: batch input: (bs, ch, w, h)
          t: time embedding: (bs, embed_dim=256)
          c: Image: (bs, 3, 64, 64) or None
        """
        x = self.down(x)
        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        if c is not None: 
            c_emb = self.c_emb_layers(c) 
        else: 
            return x + t_emb 
    
        return x + t_emb + c_emb

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim=256, scale_factor=2):
        super(UpBlock, self).__init__()

        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential(
            ResBlock(in_ch=in_ch, out_ch=in_ch),
            ResBlock(in_ch=in_ch, out_ch=out_ch), 
            ResBlock(in_ch=out_ch, out_ch=out_ch)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch),
            nn.Linear(in_features=out_ch, out_features=out_ch)
        )

        self.c_emb_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear"), 
            ResBlock(in_ch=3, out_ch=out_ch),
            ResBlock(in_ch=out_ch, out_ch=out_ch),
        )

    def forward(self, x, skip, t, c=None):

        """
        params:
          x: batch_input: (bs, ch=inp_ch/2, w, h)
          skip: from DownBLock: (bs, ch=inp_ch/2, w, h)
          t: time_embed: (bs, t_embed_dim=256)
          c: image: (bc, 3, 64, 64) or None
        """
        x = torch.cat([skip, x], dim=1)
        x = self.upsamp(x)
        x = self.up(x)
        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        
        if c is not None: 
            c_emb = self.c_emb_layers(c)
        else: 
            return x + t_emb
        
        return x + t_emb + c_emb

class AttentionBlock(nn.Module): 
    def __init__(
        self, 
        channels
    ): 
        super(AttentionBlock, self).__init__()
        self.channels = channels 

        self.norm = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=channels), 
            nn.SiLU(inplace=True)
        )

        self.q = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1) 
        self.k = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1) 
        self.v = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.out = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)  
        self.scale = channels ** -0.5 

    def forward(self, x): 

        x_norm = self.norm(x) 
        q = self.q(x_norm) 
        k = self.k(x_norm) 
        v = self.v(x_norm) 

        b, c, w, h = x.shape 

        q = q.reshape(b, c, w*h) 
        k = k.reshape(b, c, w*h) 
        v = v.reshape(b, c, w*h) 

        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=2) 

        out = torch.einsum('bij,bcj->bci', attn, v).contiguous()

        out = out.reshape(b, c, w, h) 
        out = self.out(out) 

        return x + out 

class UNet(nn.Module): 
    def __init__(
        self, 
        in_ch: int = 1, 
        t_emb_dim: int = 256, 
        base_channel: int = 64, 
        multiplier: list = [1, 2, 2, 4], 
        use_attention: bool = True, 
        time_steps: int = 1000, 
        ratio: int = 4.0 #the ratio betweent input image / condition: (256 / 64), (64 / 64) 
    ): 
        super(UNet, self).__init__() 
        self.in_ch = in_ch
        self.t_emb_dim = t_emb_dim
        self.base_channel = base_channel
        self.multiplier = multiplier 
        self.use_attention = use_attention
        self.time_steps = time_steps
        self.ratio = ratio

        # [32, 64, 128, 128] => [256, 128, 64, 32] 
        # [256, ]

        self.list_channel_down = [i * self.base_channel * 2 for i in self.multiplier] 
        self.list_channel_up = [i * self.base_channel * 4 for i in reversed(self.multiplier)] 

        self.inp = nn.Sequential(
            ResBlock(in_ch=self.in_ch, out_ch=base_channel), 
            nn.SiLU()
        )

        self.down = nn.ModuleList()
        self.up = nn.ModuleList() 

        channel = base_channel 
        
        for i in range(len(self.list_channel_down)): 
            down = DownBlock(
                in_ch = channel, 
                out_ch = self.list_channel_down[i], 
                t_emb_dim = self.t_emb_dim, 
                scale_factor = (self.ratio/(2**(i + 1)))
            ) 
            attn = AttentionBlock(channels=self.list_channel_down[i]) if self.use_attention else nn.Identity()
            self.down.append(down) 
            self.down.append(attn)
            channel = self.list_channel_down[i] 

        self.latent = nn.Sequential(
            ResBlock(in_ch=channel, out_ch=channel), 
            AttentionBlock(channels=channel) if self.use_attention else nn.Identity(), 
            ResBlock(in_ch=channel, out_ch=channel)
        )

        for i in range(len(self.list_channel_up)):
            if i == len(self.list_channel_up) - 1: 
                up = UpBlock(
                    in_ch = self.list_channel_up[i], 
                    out_ch = self.base_channel,
                    t_emb_dim = self.t_emb_dim, 
                    scale_factor = self.ratio / (2**(len(self.list_channel_up) - 2 - i + 1))
                )
                attn = AttentionBlock(channels=self.base_channel) if self.use_attention else nn.Identity()
                self.up.append(up)
                self.up.append(attn) 
            else: 
                up = UpBlock(
                    in_ch = self.list_channel_up[i], 
                    out_ch = self.list_channel_down[len(self.list_channel_up) - 2 - i],
                    t_emb_dim = self.t_emb_dim,
                    scale_factor = self.ratio / (2**(len(self.list_channel_up) - 2 - i + 1))
                )
                attn = AttentionBlock(channels=self.list_channel_down[len(self.list_channel_up) - 2 - i]) if self.use_attention else nn.Identity()
                self.up.append(up)
                self.up.append(attn)
        
        self.out = nn.Sequential( 
            nn.GroupNorm(num_groups=8, num_channels=self.base_channel),
            nn.SiLU(inplace=True),  
            nn.Conv2d(in_channels=self.base_channel, out_channels=in_ch, kernel_size=3, padding=1) 
        )
    

    def position_embeddings(self, t, channels):
        i = 1 / (10000 ** (torch.arange(start=0, end=channels, step=2) / channels)).to(t.device)
        pos_emb_sin = torch.sin(t.repeat(1, channels // 2) * i)
        pos_emb_cos = torch.cos(t.repeat(1, channels // 2) * i)
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)
        return pos_emb


    def forward(self, x, t, c=None): 
        t = self.position_embeddings(t.view(-1, 1).float().to(x.device), channels=self.t_emb_dim)
        x = self.inp(x)

        output_down = [] 

        for i in range(0, len(self.down), 2): 
            x = self.down[i](x, t, c) 
            x = self.down[i + 1](x) 
            output_down.append(x) 

        x = self.latent(x)

        for i in range(0, len(self.up), 2):
            x = self.up[i](x, output_down[-1], t, c) 
            x = self.up[i + 1](x) 
            output_down.pop()   

        x = self.out(x) 

        return x
    
class DiffusionModel(nn.Module): 
    def __init__(
        self, 
        time_steps: int = 1000,
        cfg_prob: float = 0.5,     
    ): 
        super(DiffusionModel, self).__init__() 
        self.time_steps = time_steps 
        self.cfg_prob = cfg_prob        
        
        self.denoise_net = UNet(
            in_ch = 1, 
            t_emb_dim = 256, 
            multiplier = [1, 2, 2, 4], 
            base_channel = 64, 
            use_attention = True, 
            time_steps = time_steps, 
            ratio = 1
        )

        self.scheduler = DDPMScheduler(num_train_timesteps=self.time_steps).eval() 
        self.scheduler.set_timesteps(num_inference_steps=self.time_steps)

    
    def forward(self, image: torch.Tensor, mask: torch.Tensor): 
        noise = torch.randn_like(mask, device=mask.device)
        t = torch.randint(low=0, high=self.scheduler.num_train_timesteps, size=(mask.shape[0],), device=mask.device) 
        mask_t = self.scheduler.add_noise(original_samples=mask, noise=noise, timesteps=t) 
        
        if torch.rand(1) > self.cfg_prob: 
            noise_pred = self.denoise_net.forward(x=mask_t, t=t, c=image) 
        else:
            noise_pred = self.denoise_net.forward(x=mask_t, t=t, c=None) 
        
        return noise_pred, noise
    
    
    @torch.no_grad() 
    def sample(self, image: torch.Tensor, cfg_scale: float = 4.0): 
        b, _, w, h = image.shape
        current_mask = torch.randn(size=(b, 1, w, h), device=image.device)  
        num_steps_infer = self.scheduler.timesteps
        for i in tqdm(num_steps_infer, desc="Sampling"): 
            t = (torch.ones(size=(b,), device=image.device) * i).long()

            noise_pred_cond = self.denoise_net.forward(x=current_mask, t=t, c=image)
            noise_pred_uncond = self.denoise_net.forward(x=current_mask, t=t, c=None) 
            
            # noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            noise_pred = (cfg_scale + 1) * noise_pred_cond - cfg_scale * noise_pred_uncond

            current_mask, _ = self.scheduler.step(model_output=noise_pred, timestep=i, sample=current_mask) 
            
        current_mask = (current_mask + 1.0) / 2.0
        
        return (torch.clamp(current_mask, min=0, max=1) > 0.5).long()
    


class LatentDiffusionModel(nn.Module):
    def __init__(
        self, 
        time_steps: int = 1000, 
        cfg_prob: float = 0.5, 
        vae_image_path: str = "path", 
        vae_mask_path: str = "path"
    ): 
        super(LatentDiffusionModel, self).__init__() 
        self.time_steps = time_steps 
        self.cfg_prob = cfg_prob 
        self.vae_image_module = VAEModule.load_from_checkpoint(vae_image_path) 
        self.vae_mask_module = VAEMaskModule.load_from_checkpoint(vae_mask_path) 
        self.vae_image_module.eval().freeze() 
        self.vae_mask_module.eval().freeze() 

        self.denoise_net = UNet(
            in_ch = 3, 
            t_emb_dim = 256, 
            base_channel = 64, 
            multiplier = [1, 2, 2, 4], 
            use_attention = True, 
            ratio = 1
        )

        self.scheduler = DDPMScheduler(num_train_timesteps=self.time_steps, clip_sample=True).eval() 
        self.scheduler.set_timesteps(num_inference_steps=self.time_steps) 
    
    def encode_image(self, image: torch.Tensor): 
        z_image, _ = self.vae_image_module.vae_model.encode(image)
        return z_image 
    
    def encode_mask(self, mask: torch.Tensor): 
        z_mask, _ = self.vae_mask_module.vae_model.encode(mask) 
        return z_mask 
    
    def forward(self, z_image: torch.Tensor, z_mask: torch.Tensor): 
        noise = torch.randn_like(z_mask, device=z_mask.device) 
        t = torch.randint(low=0, high=self.time_steps, size=(z_mask.shape[0],), device=z_mask.device) 
        z_mask_t = self.scheduler.add_noise(original_samples=z_mask, noise=noise, timesteps=t) 

        if torch.rand(1) > self.cfg_prob: 
            noise_pred = self.denoise_net.forward(x=z_mask_t, t=t, c=z_image) 
        else: 
            noise_pred = self.denoise_net.forward(x=z_mask_t, t=t, c=None) 

        return noise_pred, noise 

    @torch.no_grad() 
    def sample(self, z_image: torch.Tensor, cfg_scale: float = 4.0): 
        current_z_mask = torch.randn_like(z_image, device=z_image.device) 
        num_steps_infer = self.scheduler.timesteps
        for i in tqdm(num_steps_infer, desc="Sampling"): 
            t = (torch.ones(size=(z_image.shape[0],), device=z_image.device) * i).long()

            noise_pred_cond = self.denoise_net.forward(x=current_z_mask, t=t, c=z_image)
            noise_pred_uncond = self.denoise_net.forward(x=current_z_mask, t=t, c=None) 
            
            # noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            noise_pred = (cfg_scale + 1) * noise_pred_cond - cfg_scale * noise_pred_uncond

            current_z_mask, _ = self.scheduler.step(model_output=noise_pred, timestep=i, sample=current_z_mask) 
            
    
        mask_pred = self.vae_mask_module.vae_model.decode(current_z_mask) 
        mask_pred = torch.sigmoid(mask_pred) 
        mask_pred = (mask_pred > 0.5).long()
        return mask_pred 
