import torch 
from torch import nn 

import rootutils 
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True) 

from src.models.flow_matching.UNet import UNet 

class VelocityModel(nn.Module):
    def __init__(self, net: UNet): 
        super(VelocityModel, self).__init__() 
        self.net = net
    
    def forward(self, x, t, c=None): 
        velocity = self.net.forward(x=x, t=t, c=c)
        return velocity 

# net = UNet(in_ch=1, base_channel=32, multiplier=[1, 2, 4, 4], t_emb_dim=256, use_attention=True)
# velocity_model = VelocityModel(net=net)
# x = torch.randn(size=(4, 1, 256, 256)) 
# t = torch.rand(size=(4,))
# c = torch.randn(size=(4, 3, 64, 64))
# print(velocity_model(x, t, c).shape) 