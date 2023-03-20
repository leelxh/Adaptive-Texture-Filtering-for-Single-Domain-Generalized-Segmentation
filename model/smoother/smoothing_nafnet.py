# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .arch_util import LayerNorm2d
import math
from functools import partial
# from functorch import vmap
from typing import Any, Tuple, Callable, Union
import numpy as np
from matplotlib import pyplot as plt

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self._layernorm = LayerNorm(channels, eps)

    def forward(self, x):
        return self._layernorm(x)


class MetaConv2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, 
                       kernel_size : int = 3, padding : int = 1, 
                       stride : Union[int, Tuple] = 1, groups : int = 1, 
                       bias : bool = True, act_func : Callable = nn.Identity):
        super(MetaConv2d, self).__init__()
        self.linear_w = nn.Linear(1, in_channels * out_channels // groups * kernel_size * kernel_size, bias = True)
        self.linear_b = nn.Linear(1, out_channels) if bias else (lambda dumb : None)
        #print(in_channels * out_channels // groups * kernel_size * kernel_size)
        self.reshape = lambda para : para.view(out_channels, in_channels, kernel_size, kernel_size)
        self.act_func = act_func()
        self.fwd_transform = lambda lamb, img : F.conv2d(img, 
                                                        self.reshape(self.linear_w(lamb)),
                                                        self.linear_b(lamb),
                                                        stride=stride, padding=padding)        
    # def forward(self, x : torch.tensor, lamb : torch.tensor) -> torch.tensor:
    #     #print(lamb.shape, self.linear_w(lamb).shape)
    #     return self.act_func(vmap(partial(self.fwd_transform, lamb))(x))

    def forward(self, x, lamb):
        # return self.act_func(torch.stack([self.fwd_transform(lamb, item) for item in x.unbind(0)]))
        # x = torch.unsqueeze([self.fwd_transform(lamb, item) for item in x.unbind(0)], dim=0)
        x = self.fwd_transform(lamb, x)
        # x = torch.stack(x)
        return self.act_func(x)


class SimpleGate(nn.Module):

    def __init__(self):
        super(SimpleGate, self).__init__()
        self.sqrt_2 = math.sqrt(2)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = 0.5 * (1.0 + torch.erf(x2 / self.sqrt_2))
        return x1 * x2

class ReshapeToNCHW(nn.Module):
    def __init__(self):
        super(ReshapeToNCHW, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1, 1, 1)

class Sequential(nn.Sequential):
    def forward(self, x, scale=None, shift=None):
        for module in self:
            x = module(x) if scale is None else module(x, scale, shift)
        return x

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

    def forward(self, inp, scale_l=None, shift_l=None):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        if scale_l is not None: y = inp + (x * scale_l + shift_l )* self.beta
        else : y = inp + x * self.beta 
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        if scale_l is not None: return y + (x * scale_l + shift_l) * self.gamma
        return y + x * self.gamma

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.metas = nn.ModuleList()
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
            self.metas.append(MetaConv2d(chan, chan, bias=False, act_func=nn.GELU))
        
            self.decoders.append(
                Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        
    def forward(self, inp, lamb):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, meta, up, enc_skip in zip(self.decoders, self.metas, self.ups, encs[::-1]):
            x = up(x)
            #print(enc_skip.shape)
            x = x + meta(enc_skip, lamb)
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



# if __name__ == '__main__':
#
#     img_channel = 3
#     width = 32
#
#     enc_blks = [2, 2, 2, 20]
#     middle_blk_num = 2
#     dec_blks = [2, 2, 2, 2]
#
#     print('enc blks', enc_blks, 'middle blk num', middle_blk_num, 'dec blks', dec_blks, 'width' , width)
#
#     using('start . ')
#     net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
#                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
#
#     using('network .. ')
#
#     # for n, p in net.named_parameters()
#     #     print(n, p.shape)
#
#
#     inp = torch.randn((4, 3, 256, 256))
#
#     out = net(inp)
#     final_mem = using('end .. ')
#     # out.sum().backward()
#
#     # out.sum().backward()
#
#     # using('backward .. ')
#
#     # exit(0)
#
#     inp_shape = (3, 512, 512)
#
#     from ptflops import get_model_complexity_info
#
#     macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
#
#     params = float(params[:-3])
#     macs = float(macs[:-4])
#
#     print(macs, params)
#
#     print('total .. ', params * 8 + final_mem)



