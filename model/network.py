import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import copy
import sys 
sys.path.append("..") 
from models.archs.arch_util import LayerNorm2d
from models.archs.local_arch import Local_Base



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class Encoder_GCN(nn.Module):
    
    def __init__(self, img_channel=3, width=16,  enc_blk_nums=[]):
        super().__init__()
        
        
        self.conv0 = nn.Conv2d(3,16,3,1,1)
        
        self.conv_q1 = nn.Conv2d(64,64,3,1,1)
        self.conv_k1 = nn.Conv2d(64,64,3,1,1)
        self.conv1 = nn.Conv2d(64,64,3,1,1)
        
        A_base = torch.ones(64,64)
        
        self.adj1 = nn.Parameter(copy.deepcopy(A_base))
        
        self.tanh = nn.Tanh()
        self.lamda = nn.Parameter(torch.tensor(1/64))
        self.relu = nn.ReLU()
        
        self.ending = nn.Conv2d(in_channels=width, out_channels=14, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
      
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
    
    def forward(self, inp0, inp1, inp2, inp3):
        B, C, H, W = inp0.shape
        
        x0 = self.conv0(inp0)
        x1 = self.conv0(inp1)
        x2 = self.conv0(inp2)
        x3 = self.conv0(inp3)
        
        x = torch.cat((x0,x1,x2,x3),dim=1)
        
        q1 = self.conv_q1(x).mean((2,3))
        k1 = self.conv_k1(x).mean((2,3))
        
        A_bias = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A = self.adj1 + A_bias
        x = self.conv1(x)
       
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1)
        x = torch.einsum('bhwc,bcc->bhwc',[x,A]).permute(0,3,1,2)
        x = self.relu(self.lamda * x)
        
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        encs.append(x)
        
        return encs
    
class Encoder(nn.Module):
    
    def __init__(self, img_channel=3, width=16,  enc_blk_nums=[]):
        super().__init__()

        self.intro =  nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.ending = nn.Conv2d(in_channels=width, out_channels=14, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
       
        
        self.downs = nn.ModuleList()
       

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
    
    def forward(self, inp0):
        B, C, H, W = inp0.shape
        
        x = self.intro(inp0)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        encs.append(x)
        
        return encs

class End_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        
        self.conv = NAFBlock(c=in_channels)
        self.end = nn.Conv2d(in_channels,out_channels,3,1,1)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.end(x)
        
        return x

class Decoder(nn.Module):
    
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
    
        self.ending = nn.Conv2d(in_channels=width, out_channels=9, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        self.diff_conv = End_conv(9,3)
        self.spec_conv = End_conv(9,3)
        self.normal_conv = End_conv(9,2)
        self.rough_conv = End_conv(9,1)

        chan = width
        for num in enc_blk_nums:
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

    
    def forward(self,encs):
       
        x = encs[-1]
        encs = encs[:-1]
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
       
        diff = self.diff_conv(x)
        spec = self.spec_conv(x)
        normal = self.normal_conv(x)
        rough = self.rough_conv(x)
        
        return torch.cat((diff,normal,rough,spec),dim=1)

class Decoder_aniso(nn.Module):
    
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
    
        self.ending = nn.Conv2d(in_channels=width, out_channels=14, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.decoders = nn.ModuleList()
        
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        self.diff_conv = End_conv(14,3)
        self.spec_conv = End_conv(14,3)
        self.normal_conv = End_conv(14,3)
        self.tang_conv = End_conv(14,3)
        self.ax_ay_conv = End_conv(14,2)

        chan = width
        for num in enc_blk_nums:
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

    
    def forward(self,encs):
        
        x = encs[-1]
        encs = encs[:-1]
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        
        diff = self.diff_conv(x)
        
        spec = self.spec_conv(x)
        normal = self.normal_conv(x)
        tangent = self.tang_conv(x)
        ax_ay = self.ax_ay_conv(x)
        
        return torch.cat((diff,normal,ax_ay, spec, tangent),dim=1)



if __name__ == '__main__':

    img_channel = 3
    width = 64
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    inp0 = torch.randn(1,3,256,256).cuda()
    net = Encoder_GCN(img_channel=img_channel, width=width, enc_blk_nums=enc_blks).cuda()
    y = net(inp0, inp0, inp0,inp0)
    
    model = Decoder(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()
    z = model(y)
    
