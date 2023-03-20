import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torchvision.models as models
import cv2
from torch.autograd import Variable
from .model_util import *
from .seg_model import DeeplabMulti
from .smoother import NAFNet
from scipy.special import erfinv
from scipy.ndimage.filters import gaussian_filter
from math import sqrt
import scipy.stats as stats

pspnet_specs = {
    'n_classes': 19,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}
'''
Sequential blocks
'''


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=16):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class BoundaryMapping5(nn.Module):
    def __init__(self, num_input, num_output, kernel_sz=None, stride=None, padding=None):
        super(BoundaryMapping5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 8),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input // 8, num_output, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        so_output = self.conv1(x)
        so_output = self.conv1_1(so_output)
        so_output = self.conv1_2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv2(so_output)
        so_output = self.conv2_1(so_output)
        so_output = self.conv2_2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv3_1(so_output)
        so_output = self.conv3(so_output)
        so_output = self.upsample2(so_output)

        return so_output


class BoundaryMapping2(nn.Module):
    def __init__(self, num_input, num_output, kernel_sz=None, stride=None, padding=None):
        super(BoundaryMapping2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 8),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input // 8, num_output, kernel_size=3, padding=1, bias=False)
        )
       
    def forward(self, x_1):
        so_output = self.conv1(x_1)
        so_output = self.conv1_1(so_output)
        so_output = self.conv1_2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv2(so_output)
        so_output = self.conv2_1(so_output)
        so_output = self.conv2_2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv3_1(so_output)
        so_output = self.conv3(so_output)

        return so_output


class BoundaryMapping3(nn.Module):
    def __init__(self, num_input, num_output, kernel_sz=None, stride=None, padding=None):
        super(BoundaryMapping3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_input // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 8),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input // 8, num_output, kernel_size=3, padding=1, bias=False)
        )
        self.downsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    def forward(self, x_2):
        so_output = self.conv1(x_2)
        so_output = self.conv1_1(so_output)
        so_output = self.conv1_2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv2(so_output)
        so_output = self.conv2_1(so_output)
        so_output = self.conv2_2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv3_1(so_output)
        so_output = self.conv3(so_output)
        so_output = self.upsample2(so_output)

        return so_output


class Smooth_Etract(nn.Module):
    def __init__(self, num_input=3, num_output=64):
        super(Smooth_Etract, self).__init__()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, num_output, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_output),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_output, num_output, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_output),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_output, num_output, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_output),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Rest_Etract(nn.Module):
    def __init__(self, num_input=3, num_output=64):
        super(Rest_Etract, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv6_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv6_2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv6_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv7_3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x_1 = self.conv4(x)

        x_2 = self.upsample1(x_1)
        x_2 = self.conv5(x_2)
        x_2 = self.conv5_1(x_2)
        x_2 = self.conv5_2(x_2)
        x_2 = self.conv5_3(x_2)

        x_3 = self.upsample1(x_2)
        x_3 = self.conv6(x_3)
        x_3 = self.conv6_1(x_3)
        x_3 = self.conv6_2(x_3)
        x_3 = self.conv6_3(x_3)

        x_4 = self.conv7(x_3)
        x_4 = self.conv7_1(x_4)
        x_4 = self.conv7_2(x_4)
        x_4 = self.conv7_3(x_4)

        return x_1, x_2, x_3, x_4


class SNR(nn.Module):

    def __init__(self, input_channels, reduction_ratio=16):
        super(SNR, self).__init__()
        self.input_channels = input_channels
        self.reduction_ratio = reduction_ratio
        self.InstanceNorm = nn.InstanceNorm2d(self.input_channels, affine=True)
        self.SE = ChannelSELayer(self.input_channels, self.reduction_ratio)


    def forward(self, f):
        ff = self.InstanceNorm(f)
        r = f - ff
        r_plus = self.SE(r)
        ff_plus = ff + r_plus
        return ff_plus


class Resize(nn.Module):
    def __init__(self, a, b, c, d):
        super(Resize, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, x):
        x = x.resize(self.a, self.b, self.c, self.d)
        return x


class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.n_classes = pspnet_specs['n_classes']

        Seg_Model = DeeplabMulti(num_classes=self.n_classes)

        self.layer0 = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu)

        self.layer1 = Seg_Model.layer1
        self.layer2 = Seg_Model.layer2
        self.layer3 = Seg_Model.layer3
        self.layer4 = Seg_Model.layer4

        self.classifer3 = Seg_Model.layer5
        self.classifer4 = Seg_Model.layer6
        self.IN_0 = Rest_Etract()
        self.IN_1 = SNR(256)
        self.IN_2 = SNR(512)
        self.IN_3 = SNR(1024)
        self.IN_4 = SNR(2048)
        # self.fb_1 = BoundaryMapping2(256, 1)
        # self.fb_2 = BoundaryMapping3(512, 1)
        # self.fb_3 = BoundaryMapping3(1024, 1)
        # self.fb_4 = BoundaryMapping5(2048, 1)

        self.smooth_extract = Smooth_Etract()

        self.score_final = nn.Conv2d(4, 1, 1)  # map
        self.smooth_p = nn.Linear(128, 1)
        self.smoother = NAFNet(img_channel=3, width=32, middle_blk_num=2, enc_blk_nums=[2, 2, 2, 2],
                               dec_blk_nums=[2, 2, 2, 2])
        self.downsample4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False)
        self.downsample8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=False)

    def forward(self, x, aux=None, original=None):
        global ratio
        image_nochange = x
        x = (x + 1) / 2

        x_smooth = self.smooth_extract(x)
        # print('x_smooth', x_smooth.shape)
        x_smooth_mean = x_smooth.mean(dim=[2, 3], keepdim=True)
        x_smooth_sigm = (x_smooth.std(dim=[2, 3], keepdim=True)) ** 2

        smooth_param = torch.cat((x_smooth_mean, x_smooth_sigm), dim=1)
        smooth_param = torch.squeeze(smooth_param, dim=3)
        smooth_param = torch.squeeze(smooth_param, dim=2)
        smooth_param = self.smooth_p(smooth_param)

        ratio = torch.sigmoid(smooth_param)
        smooth_param_2 = ratio

        if aux is not None:
            ratio = torch.randn(1).cuda() + smooth_param_2
            if ratio <= 0:
                ratio = smooth_param_2
            if ratio >= smooth_param_2 + 1.5:
                ratio = smooth_param_2

        x_x = self.smoother(x, ratio)
        x_x = x_x * 2 - 1
        x_rest = image_nochange - x_x
        x_rest_1, x_rest_2, x_rest_3, x_rest_4 = self.IN_0(x_rest)

        low = self.layer0(x_x)
        low = low + x_rest_1

        x_1 = self.layer1(low)
        x_1 = self.IN_1(x_1)
        # boundary_1 = self.fb_1(x_1)
        # boundary_1_a = self.downsample4(boundary_1)

        x_rest_2 = x_rest_2 # * boundary_1_a
        x_2 = self.layer2(x_1 + x_rest_2)
        x_2 = self.IN_2(x_2)
        # boundary_2 = self.fb_2(x_2)
        # boundary_2_a = self.downsample8(boundary_2)

        x_rest_3 = x_rest_3 # * boundary_2_a
        x_3 = self.layer3(x_2 + x_rest_3)
        x_3 = self.IN_3(x_3)
        # boundary_3 = self.fb_3(x_3)
        # boundary_3_a = self.downsample8(boundary_3)

        x_rest_4 = x_rest_4 # * boundary_3_a
        x_4 = self.layer4(x_3 + x_rest_4)
        x_4 = self.IN_4(x_4)
        # boundary_4 = self.fb_4(x_4)

        x3 = self.classifer3(x_3)
        x4 = self.classifer4(x_4)

        # fusecat = torch.cat((boundary_4, boundary_3, boundary_2, boundary_1), dim=1)
        # fuse = self.score_final(fusecat)

        # results = [boundary_4, boundary_3, boundary_2, boundary_1, fuse]
        # results = [torch.sigmoid(r) for r in results]

        if aux is not None:
            x_rest = x_rest * 2 - 1
            return x_x, x_rest, x3, x4
        else:
            return x_x, x3, x4

    def get_1x_lr_params_NOscale(self):
        # b = []
        # # b.append(self.layer_0)
        # b.append(self.layer0)
        # b.append(self.layer1)
        # b.append(self.layer2)
        # b.append(self.layer3)
        # b.append(self.layer4)

        # for i in range(len(b)):
        #     for j in b[i].modules():
        #         jj = 0
        #         for k in j.parameters():
        #             jj += 1
        #             if k.requires_grad:
        #                 yield k
        pass
    def get_10x_lr_params(self):
        b = []
        b.append(self.classifer3.parameters())
        b.append(self.classifer4.parameters())
        b.append(self.smooth_p.parameters())
        b.append(self.smooth_extract.parameters())
        b.append(self.IN_0.parameters())
        b.append(self.IN_1.parameters())
        b.append(self.IN_2.parameters())
        b.append(self.IN_3.parameters())
        b.append(self.IN_4.parameters())
        # b.append(self.fb_3.parameters())
        # b.append(self.fb_2.parameters())
        # b.append(self.fb_1.parameters())
        # b.append(self.fb_4.parameters())
        # b.append(self.sp_1.parameters())
        # b.append(self.sp_2.parameters())
        # b.append(self.sp_3.parameters())
        # b.append(self.sp_4.parameters())
        b.append(self.score_final.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]

