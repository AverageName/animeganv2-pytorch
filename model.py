import torch
from torch import nn
import torch.nn.functional as F


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        
        pad_layer = {
            "zero":    nn.ZeroPad2d,
            "same":    nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError
            
        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch*expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        
        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out

    
class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.block_a = nn.Sequential(
            ConvNormLReLU(3,  32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0,1,0,1)),
            ConvNormLReLU(64, 64)
        )
        
        self.block_b = nn.Sequential(
            ConvNormLReLU(64,  128, stride=2, padding=(0,1,0,1)),            
            ConvNormLReLU(128, 128)
        )
        
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )    
        
        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64,  64),
            ConvNormLReLU(64,  32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)
        
        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out

class ConvLayerNormLeakyReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, spatial, spec_norm, norm=None):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                  stride, kernel_size // 2, bias=False)
        
        if spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm is not None:
            self.norm = nn.LayerNorm([out_channels, *spatial])
        else:
            self.norm = nn.Identity()
        
            
    def forward(self, x):
        return F.leaky_relu(self.norm(self.conv(x)), 0.2)

        
class Discriminator(nn.Module):

    def __init__(self, in_shape, n_dis, spec_norm):
        super().__init__()

        out_channel = in_shape[0] // 2
        in_channel = in_shape[0]

        spatial = in_shape[1:]

        self.first_block = ConvLayerNormLeakyReLU(in_channel, out_channel, 3, 1, spatial, spec_norm)
        in_channel = out_channel
        self.blocks = []

        for i in range(1, n_dis):
            self.blocks.append(ConvLayerNormLeakyReLU(in_channel, out_channel * 2, 3, 2, spatial, spec_norm))
            spatial = [spatial[0] // 2, spatial[1] // 2]

            self.blocks.append(ConvLayerNormLeakyReLU(out_channel * 2, out_channel * 4, 3, 1, spatial, spec_norm, norm='layer'))
            in_channel = out_channel * 4
            out_channel = out_channel * 2
          
        self.last_block = ConvLayerNormLeakyReLU(in_channel, out_channel * 2, 3, 1, spatial, spec_norm, norm='layer')
        self.last_conv = nn.Conv2d(out_channel * 2, 1, 3, 1, bias=False)
        if spec_norm:
            self.last_conv = nn.utils.spectral_norm(self.last_conv)
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        out = self.first_block(x)
        out = self.blocks(out)
        out = self.last_block(out)
        out = self.last_conv(out)

        return out
        