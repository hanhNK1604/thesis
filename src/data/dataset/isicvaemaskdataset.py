import torch 
import os 
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image 

import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class ISICVAEMaskDataset(Dataset): 
    def __init__(
        self, 
        image_size: int = 256, 
        data_dir: str = "data" 
    ):
        super(ISICVAEMaskDataset, self).__init__()
        self.image_size = image_size
        self.data_dir = data_dir 

        self.data_path = os.path.join(self.data_dir, "isic", "labels") 
        
        self.image_files = os.listdir(self.data_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((self.image_size, self.image_size), antialias=True),
        ])

    def __len__(self): 
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.image_files[index])
        
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        
        return image
        
# data_dir = "/home/ubuntu/thesis/data" 
# dataset = ISICVAEMaskDataset(image_size=256, data_dir=data_dir) 

# print(dataset.__getitem__(0).max())
# print(dataset.__getitem__(0).min())
        
# print(dataset.__getitem__(1))
# print(dataset.__getitem__(1).shape)
# print(len(dataset))

