from torch import nn
import torch

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
        """
        params:
          x: batch input: (bs, ch, w, h)
        """
        
        out = self.conv(x) 
        res = self.res_conv(x) 

        return out + res 
    


# x = torch.rand(size=(4, 1, 32, 32))
# net = ResBlock(in_ch=1, out_ch=64)
# print(net(x).shape)
