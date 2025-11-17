import torch 
import os
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image 

import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class ClinicFMDataset(Dataset): 
    def __init__(
        self, 
        data_dir: str = "data"
    ): 
        super(ClinicFMDataset, self).__init__() 
        self.data_dir = data_dir

        self.data_path_image = os.path.join(self.data_dir, "cvc-clinic", "images") 
        self.data_path_label = os.path.join(self.data_dir, "cvc-clinic", "labels")

        self.image_files = os.listdir(self.data_path_image)

        self.transform_image = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
        ])

        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,)) 
        ])
    
    def __len__(self): 
        return len(self.image_files) 
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_path_image, self.image_files[index])
        image = Image.open(img_path).convert("RGB")
        image = self.transform_image(image)

        label_path = os.path.join(self.data_path_label, self.image_files[index]) 
        label = Image.open(label_path).convert("L")
        label = self.transform_label(label) 


        # print(img_path) 
        # print(label_path)

        return image, label 
    

# data_dir = "/home/ubuntu/thesis/data" 

# dataset = ClinicFMDataset(data_dir=data_dir) 

# x = dataset.__getitem__(10) 

# print(x.shape) 
 
# print(x)

# # print(image.max(), image.min()) 
# print(x.max(), x.min()) 
 
# for i in label