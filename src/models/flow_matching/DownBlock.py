from torch import nn
import torch
import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.flow_matching.ResBlock import ResBlock

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

#test
# x = torch.rand(size=(32, 64, 128, 128))
# t = torch.rand(size=(32, 256))
# c = torch.rand(size=(32, 3, 64, 64))
# net = DownBlock(in_ch=64, out_ch=128, scale_factor=1)
# print(net(x, t, c).shape)