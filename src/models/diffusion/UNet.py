import torch
from torch import nn

import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.flow_matching.ResBlock import ResBlock 
from src.models.flow_matching.DownBlock import DownBlock 
from src.models.flow_matching.UpBlock import UpBlock 
from src.models.flow_matching.AttentionBlock import AttentionBlock 


class UNet(nn.Module): 
    def __init__(
        self, 
        in_ch: int = 1, 
        t_emb_dim: int = 256, 
        base_channel: int = 32, 
        multiplier: list = [1, 2, 4, 4], 
        use_attention: bool = True, 
        ratio: int = 4.0 #the ratio betweent input image / condition: (256 / 64), (64 / 64) 
    ): 
        super(UNet, self).__init__() 
        self.in_ch = in_ch
        self.t_emb_dim = t_emb_dim
        self.base_channel = base_channel
        self.multiplier = multiplier 
        self.use_attention = use_attention
        self.ratio = ratio

        self.list_channel_down = [i * self.base_channel * 2 for i in self.multiplier] 
        self.list_channel_up = [i * self.base_channel * 4 for i in reversed(self.multiplier)] 

        self.inp = nn.Sequential(
            ResBlock(in_ch=self.in_ch, out_ch=base_channel), 
            nn.SiLU()
        )

        self.down = nn.ModuleList()
        self.up = nn.ModuleList() 

        channel = base_channel 
        
        for i in range(len(self.list_channel_down)): 
            down = DownBlock(
                in_ch = channel, 
                out_ch = self.list_channel_down[i], 
                t_emb_dim = self.t_emb_dim, 
                scale_factor = (self.ratio/(2**(i + 1)))
            ) 
            attn = AttentionBlock(channels=self.list_channel_down[i]) if self.use_attention else nn.Identity()
            self.down.append(down) 
            self.down.append(attn)
            channel = self.list_channel_down[i] 

        self.latent = nn.Sequential(
            ResBlock(in_ch=channel, out_ch=channel), 
            AttentionBlock(channels=channel) if self.use_attention else nn.Identity(), 
            ResBlock(in_ch=channel, out_ch=channel)
        )

        for i in range(len(self.list_channel_up)):
            if i == len(self.list_channel_up) - 1: 
                up = UpBlock(
                    in_ch = self.list_channel_up[i], 
                    out_ch = self.base_channel,
                    t_emb_dim = self.t_emb_dim, 
                    scale_factor = self.ratio / (2**(len(self.list_channel_up) - 2 - i + 1))
                )
                attn = AttentionBlock(channels=self.base_channel) if self.use_attention else nn.Identity()
                self.up.append(up)
                self.up.append(attn) 
            else: 
                up = UpBlock(
                    in_ch = self.list_channel_up[i], 
                    out_ch = self.list_channel_down[len(self.list_channel_up) - 2 - i],
                    t_emb_dim = self.t_emb_dim,
                    scale_factor = self.ratio / (2**(len(self.list_channel_up) - 2 - i + 1))
                )
                attn = AttentionBlock(channels=self.list_channel_down[len(self.list_channel_up) - 2 - i]) if self.use_attention else nn.Identity()
                self.up.append(up)
                self.up.append(attn)
        
        self.out = nn.Sequential( 
            nn.GroupNorm(num_groups=8, num_channels=self.base_channel),
            nn.SiLU(inplace=True),  
            nn.Conv2d(in_channels=self.base_channel, out_channels=in_ch, kernel_size=3, padding=1) 
        )
        
    def position_embeddings(self, t, channels):
        i = 1 / (10000 ** (torch.arange(start=0, end=channels, step=2) / channels)).to(t.device)
        pos_emb_sin = torch.sin(t.repeat(1, channels // 2) * i)
        pos_emb_cos = torch.cos(t.repeat(1, channels // 2) * i)
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)
        return pos_emb

    def forward(self, x, t, c=None): 
        t = t.view(-1, 1).float().to(x.device)
        t = self.position_embeddings(t, self.t_emb_dim) 
        x = self.inp(x)

        output_down = [] 

        for i in range(0, len(self.down), 2): 
            x = self.down[i](x, t, c) 
            x = self.down[i + 1](x) 
            output_down.append(x) 

        x = self.latent(x)

        for i in range(0, len(self.up), 2):
            x = self.up[i](x, output_down[-1], t, c) 
            x = self.up[i + 1](x) 
            output_down.pop()   

        x = self.out(x) 

        return x


# net = UNet(in_ch=3, base_channel=64, multiplier=[1, 2, 2, 4], t_emb_dim=256, use_attention=True, ratio=1.0)

# x = torch.rand(size=(4, 3, 64, 64))
# t = torch.rand(size=(4,))
# c = torch.rand(size=(4, 3, 64, 64))
# print(net(x, t, c).shape)

# model = net
# param_size = 0
# for param in model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (param_size + buffer_size) / 1024**2
# print('model size: {:.3f}MB'.format(size_all_mb))