from PIL import Image 
from torchvision import transforms 
from torchvision.utils import make_grid 
import torch
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
 
image = Image.open("/home/ubuntu/thesis/data/isic/labels/ISIC_0000062.jpg")
transform = transforms.Compose([
    transforms.ToTensor(), 
    # transforms.Resize((64, 64)) 
])

num_timesteps = 10

image = transform(image) 
image = torch.unsqueeze(image, dim=0)
print((image > 0.5).long().sum())
scheduler = DDPMScheduler(num_train_timesteps=num_timesteps, schedule="linear_beta") 
scheduler.set_timesteps(num_inference_steps=num_timesteps) 
timesteps = torch.tensor([i for i in range(num_timesteps)], dtype=torch.long) 

image = torch.cat([image] * num_timesteps, dim=0) 
noise = torch.randn_like(image) 
image_t = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps) 

image_t = make_grid(image_t, nrow=10) 
image_t = transforms.ToPILImage()(image_t) 
image_t.save("/home/ubuntu/thesis/src/models/diffusion/image.png")
