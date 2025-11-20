import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.fm_latent_module import FlowMatchingLatentModule
from src.models.diffusion_module import DiffusionModule
from src.models.latent_diffusion_module import LatentDiffusionModule 
import streamlit as st 
from PIL import Image
import torch 
from torchvision import transforms 
import os 

# checkpoint = '/workspace/thesis/checkpoints/fm_latent/isic/epoch=149-step=125400.ckpt'
# module = FlowMatchingLatentModule.load_from_checkpoint(
#     checkpoint_path = checkpoint, 
#     vae_image_path = "/workspace/thesis/checkpoints/vae_image/isic/epoch=249-step=209000.ckpt",
#     vae_mask_path = "/workspace/thesis/checkpoints/vae_mask/isic/epoch=99-step=83600.ckpt" 
# ) 
# module.eval().freeze()

# checkpoint = '/workspace/thesis/checkpoints/fm_latent/clinic/epoch=439-step=54121.ckpt'
# module = FlowMatchingLatentModule.load_from_checkpoint(
#     checkpoint_path = checkpoint, 
#     vae_image_path = "/workspace/thesis/checkpoints/vae_image/clinic/epoch=249-step=30750.ckpt",
#     vae_mask_path = "/workspace/thesis/checkpoints/vae_mask/clinic/epoch=199-step=24600.ckpt" 
# ) 
# module.eval().freeze()

# def result(image_path): 
#     image = Image.open(image_path)
#     transform = transforms.Compose([
#         transforms.ToTensor(), 
#         transforms.Resize((256, 256)), 
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
#     filename = os.path.basename(image_path)      
#     name = os.path.splitext(filename)[0]   

#     image = transform(image).to("cuda")
#     image = torch.unsqueeze(image, dim=0) 
#     z_image, _ = module.vae_image_module.vae_model.encode(image)
#     samples = [] 
#     for i in range(5): 
#         sample = module.sample(input_size=z_image.shape, z_image=z_image) 
#         samples.append(torch.unsqueeze(sample, dim=0)) 
#         sample = sample[0] 
#         sample = transforms.ToPILImage()(sample.cpu().float())  
#         sample.save(f"/workspace/thesis/result/isic/{name}_{i}.png")



#     res = torch.cat(samples, dim=0) 
#     res = torch.mean(res.float(), dim=0)
#     res = (res > 0.5).long()[0]
#     res = transforms.ToPILImage()(res.cpu().float())
#     res.save(f"/workspace/thesis/result/isic/{name}_ensemble.png")

#     res_samples = torch.cat(samples, dim=0).float().to("cuda")   # [5,1,256,256]
#     var = torch.var(res_samples, dim=0, unbiased=False)  # [1, H, W]
#     var_image = var[0]                                
#     var_image = (var_image - var_image.min()) / (var_image.max() - var_image.min() + 1e-8)  # normalize 0–1

#     var_pil = transforms.ToPILImage()(var_image.cpu())
#     var_pil.save(f"/workspace/thesis/result/isic/{name}_variance.png")

# result('/workspace/thesis/data/isic/images/ISIC_0000095.jpg')

# checkpoint_diff = '/workspace/thesis/logs/train/runs/2025-11-18_04-31-33/thesis-diffusion/pyih13v4/checkpoints/epoch=249-step=209000.ckpt'
# module_diff = DiffusionModule.load_from_checkpoint(checkpoint_path=checkpoint_diff) 
# module_diff.eval().freeze() 

# checkpoint_diff = '/workspace/thesis/logs/train/runs/2025-11-18_10-35-55/thesis-diffusion/hnppzcb7/checkpoints/epoch=249-step=30751.ckpt'
# module_diff = DiffusionModule.load_from_checkpoint(checkpoint_path=checkpoint_diff) 
# module_diff.eval().freeze() 


# def result_diffusion(image_path): 
#     image = Image.open(image_path) 
#     transform = transforms.Compose([
#         transforms.ToTensor(), 
#         transforms.Resize((64, 64)), 
#     ]) 

#     filename = os.path.basename(image_path)      
#     name = os.path.splitext(filename)[0]  

#     image = transform(image).to("cuda") 
#     image = image * 2.0 - 1.0
#     image = torch.unsqueeze(image, dim=0) 
#     samples = [] 
#     for i in range(5): 
#         sample = module_diff.diffusion_model.sample(image=image, cfg_scale=4.0)
#         sample = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)(sample)
#         samples.append(torch.unsqueeze(sample, dim=0)) 
#         sample = sample[0]
#         sample = transforms.ToPILImage()(sample.cpu().float()) 
#         sample.save(f"/workspace/thesis/result/isic/diff_{name}_{i}.png")
    
#     res = torch.cat(samples, dim=0) 
#     res = torch.mean(res.float(), dim=0)
#     res = (res > 0.5).long()[0]
#     res = transforms.ToPILImage()(res.cpu().float())
#     res.save(f"/workspace/thesis/result/isic/diff_{name}_ensemble.png")

#     res_samples = torch.cat(samples, dim=0).float().to("cuda")   # [5,1,256,256]
#     var = torch.var(res_samples, dim=0, unbiased=False)  # [1, H, W]
#     var_image = var[0]                                
#     var_image = (var_image - var_image.min()) / (var_image.max() - var_image.min() + 1e-8)  # normalize 0–1

#     var_pil = transforms.ToPILImage()(var_image.cpu())
#     var_pil.save(f"/workspace/thesis/result/isic/diff_{name}_variance.png")
# result_diffusion('/workspace/thesis/data/isic/images/ISIC_0000186.jpg')


checkpoint_latent_diff = '' 
module_latent_diff = LatentDiffusionModule.load_from_checkpoint(checkpoint_latent_diff) 
module_latent_diff.eval().freeze() 

def result_latent_diffusion(image_path): 
    image = Image.open(image_path) 
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
    ])

    filename = os.path.basename(image_path)      
    name = os.path.splitext(filename)[0]  

    image = transform(image).to("cuda") 
    image = torch.unsqueeze(image, dim=0) 
    z_image, _ = module_latent_diff.diffusion_model.vae_image_module.vae_model.encode(image)
    samples = [] 
    for i in range(5): 
        sample = module_latent_diff.diffusion_model.sample(z_image=z_image, cfg_scale=4.0) 
        samples.append(torch.unsqueeze(sample, dim=0))
        sample = sample[0] 
        sample = transforms.ToPILImage(sample.cpu().float()) 
        sample.save(f"/workspace/thesis/result/isic/diff_latent_{name}_{i}.png")
        
    res = torch.cat(samples, dim=0) 
    res = torch.mean(res.float(), dim=0)
    res = (res > 0.5).long()[0]
    res = transforms.ToPILImage()(res.cpu().float())
    res.save(f"/workspace/thesis/result/isic/diff_latent_{name}_ensemble.png")

    res_samples = torch.cat(samples, dim=0).float().to("cuda")   # [5,1,256,256]
    var = torch.var(res_samples, dim=0, unbiased=False)  # [1, H, W]
    var_image = var[0]                                
    var_image = (var_image - var_image.min()) / (var_image.max() - var_image.min() + 1e-8)  # normalize 0–1

    var_pil = transforms.ToPILImage()(var_image.cpu())
    var_pil.save(f"/workspace/thesis/result/isic/diff_latent_{name}_variance.png")