import torch 
import os
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image 

import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class DiffusionDataset(Dataset): 
    def __init__(
        self, 
        data_dir: str = "data/isic"
    ): 
        super(DiffusionDataset, self).__init__() 
        self.data_dir = data_dir

        self.data_path_image = os.path.join(self.data_dir, "images") 
        self.data_path_label = os.path.join(self.data_dir, "labels")

        self.image_files = os.listdir(self.data_path_image)

        self.transform_image_original = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))    
        ])
        self.transform_image_resize = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((64, 64,), transforms.InterpolationMode.BILINEAR),
        ])

        self.transform_label_original = transforms.Compose([
            transforms.ToTensor()             
        ])
        self.transform_label_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64), transforms.InterpolationMode.NEAREST),
        ])
    
    def __len__(self): 
        return len(self.image_files) 
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_path_image, self.image_files[index])
        image = Image.open(img_path).convert("RGB")
        image_original = self.transform_image_original(image) 
        image_resize = self.transform_image_resize(image)

        label_path = os.path.join(self.data_path_label, self.image_files[index]) 
        label = Image.open(label_path).convert("L")
        label_original = self.transform_label_original(label) 
        label_resize = self.transform_label_resize(label) 

        return image_resize, label_resize, image_original, label_original 

    

# data_dir = "/home/ubuntu/thesis/data/isic" 

# dataset = DiffusionDataset(data_dir=data_dir) 

# image, label, _, _ = dataset.__getitem__(10) 

# print(image.shape, label.shape) 
# print(image) 
# print(label)

# print(image.max(), image.min()) 
# print(label.max(), label.min()) 
 
# for i in label