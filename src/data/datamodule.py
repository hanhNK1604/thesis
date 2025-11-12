from lightning import LightningDataModule 
import torch
from torch.utils.data import DataLoader, Dataset, random_split 
import rootutils 

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 
from src.data.dataset.isicvaedataset import ISICVAEDataset
from src.data.dataset.isicvaemaskdataset import ISICVAEMaskDataset

class Datamodule(LightningDataModule): 
    def __init__(
        self, 
        dataset: Dataset, 
        train_val_test_split: list,
        batch_size: int, 
        num_workers: int, 
        pin_memory: bool
    ): 
        super(Datamodule, self).__init__()
        self.dataset = dataset
        self.train_val_test_split = train_val_test_split 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory 

        self.data_train = None 
        self.data_test = None 
        self.data_val = None

    def setup(self, stage: str = None): 
        if not self.data_train and not self.data_val and not self.data_test: 
            self.data_train, self.data_val, self.data_test = random_split(dataset=self.dataset, lengths=self.train_val_test_split, generator=torch.Generator().manual_seed(40))

    def train_dataloader(self): 
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def test_dataloader(self): 
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def val_dataloader(self): 
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
# data_dir = "/home/ubuntu/thesis/data" 
# data_set = ISICVAEMaskDataset(image_size=256, data_dir=data_dir) 

# data_module = Datamodule(dataset=data_set, train_val_test_split=[3344, 100, 250], batch_size=32, num_workers=2, pin_memory=False)
# data_module.setup()
# print(len(data_module.train_dataloader()))