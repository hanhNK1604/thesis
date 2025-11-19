from PIL import Image 
from torchvision.utils import make_grid 
from torchvision import transforms 
import torch 
import os 

def plot_data(list_paths):  
    images = [] 
    for path in list_paths: 
        image = Image.open(path) 
        image = transforms.ToTensor()(image) 
        images.append(torch.unsqueeze(image, dim=0)) 
    images = torch.cat(images, dim=0) 
    images = make_grid(images, nrow=3) 
    images = transforms.ToPILImage()(images) 
    images.save('/home/ubuntu/thesis/result/clinic_label.png')


list_paths = [
    '/home/ubuntu/thesis/data/cvc-clinic/labels/1.png',
    '/home/ubuntu/thesis/data/cvc-clinic/labels/10.png',
    '/home/ubuntu/thesis/data/cvc-clinic/labels/20.png',
    '/home/ubuntu/thesis/data/cvc-clinic/labels/30.png',
    '/home/ubuntu/thesis/data/cvc-clinic/labels/40.png',
    '/home/ubuntu/thesis/data/cvc-clinic/labels/50.png',
    '/home/ubuntu/thesis/data/cvc-clinic/labels/60.png',
    '/home/ubuntu/thesis/data/cvc-clinic/labels/70.png',
    '/home/ubuntu/thesis/data/cvc-clinic/labels/80.png',
]

plot_data(list_paths=list_paths)