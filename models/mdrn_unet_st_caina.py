import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
from skimage.transform import resize
from torch.utils import model_zoo

from models.style_modules import ada_in, ada_in_max, ada_in_cluster, PatchTransferCluster


# Extract the HW center of a 4-D tensor BCHW
def extractCenter(x, pad, pad2=0):
    if (pad2 == 0):
        return x[:, :, pad:-pad, pad:-pad, ]
    else:
        return x[:, :, pad:-pad2, pad:-pad2, ]


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, strides=2, kernel_size=4, activ='relu', norm=True,
                 use_dropout=True, padding=1, drop_rate=0.5):
        super(EncoderBlock, self).__init__()

        self.use_dropout = use_dropout
        self.norm = norm

        self.activate = activation_func(activ)
        self.conv = nn.Conv2d(input_channels, output_channels, stride=2, kernel_size=4, padding_mode='zeros',
                              padding=padding)
        self.norm_layer = nn.BatchNorm2d(output_channels)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm_layer(x)
        x = self.activate(x)
        if self.use_dropout:
            x = self.drop(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=2, kernel_size=4, activ='relu', norm=True,
                 use_dropout=True, padding=1, drop_rate=0.5):
        super(DecoderBlock, self).__init__()

        self.use_dropout = use_dropout
        self.norm = norm

        self.activate = activation_func(activ)
        self.deconv = nn.ConvTranspose2d(input_channels, output_channels, stride=2, kernel_size=4, padding_mode='zeros',
                                         padding=padding)
        self.norm_layer = nn.BatchNorm2d(output_channels)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, net, s1, s2, s3):
        dc = []
        #         print(s1.shape)
        #         print(s2.shape)
        #         print(s3.shape)
        if net is not None: dc.append(net)
        if s1 is not None: dc.append(s1)
        if s2 is not None: dc.append(s2)
        if s3 is not None: dc.append(s3)

        x = torch.cat(dc, 1)
        x = self.deconv(x)
        if self.norm:
            x = self.norm_layer(x)
        x = self.activate(x)
        if self.use_dropout:
            x = self.drop(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, padding_mode='zeros', padding=1,
                               dilation=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride=stride, padding_mode='zeros',
                               padding=1)
        self.conv3 = nn.Conv2d(output_channels, output_channels, kernel_size, stride=1, padding_mode='zeros', padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, output_padding=1):
        super(DeconvBlock, self).__init__()
        # self.conv1 = nn.ConvTranspose2d(input_channels, input_channels, kernel_size, stride=stride,
        #                                 padding_mode='zeros', padding=1, output_padding=output_padding)
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size, padding_mode='zeros', padding=1)
        self.pxsf = nn.PixelShuffle(stride)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, padding_mode='zeros', padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pxsf(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.config = config
        in_channels, out_channels = self.config['in_channels'], self.config['out_channels']
        feat = self.config['feat']
        activation = self.config['activation']
        kernel_size = self.config['kernel_size']
        num_layers = self.config['num_layers']
        img_channels = self.config['img_channels']
        self.factor = self.config['downsample_factor']
        depth = self.config['depth']
        self.batch_size = self.config['batch_size']
        self.img_shape = self.config["img_shape"]

        self.shortcut = nn.Identity()
        self.upsample = nn.Upsample(scale_factor=self.factor, mode='nearest')

        self.s1_conv1 = EncoderBlock(img_channels + 1, depth, norm=False, use_dropout=False)
        self.s1_conv2 = EncoderBlock(depth, depth * 2, use_dropout=False)
        self.s1_conv3 = EncoderBlock(depth * 2, depth * 4, use_dropout=False)
        self.s1_conv4 = EncoderBlock(depth * 4, depth * 8, use_dropout=False)
        self.s1_conv5 = EncoderBlock(depth * 8, depth * 8)
        self.s1_conv6 = EncoderBlock(depth * 8, depth * 8)

        self.s2_conv1 = EncoderBlock(img_channels + 1, depth, norm=False, use_dropout=False)
        self.s2_conv2 = EncoderBlock(depth, depth * 2, use_dropout=False)
        self.s2_conv3 = EncoderBlock(depth * 2, depth * 4, use_dropout=False)
        self.s2_conv4 = EncoderBlock(depth * 4, depth * 8, use_dropout=False)
        self.s2_conv5 = EncoderBlock(depth * 8, depth * 8)
        self.s2_conv6 = EncoderBlock(depth * 8, depth * 8)

        self.s3_conv1 = EncoderBlock(img_channels + 1, depth, norm=False, use_dropout=False)
        self.s3_conv2 = EncoderBlock(depth, depth * 2, use_dropout=False)
        self.s3_conv3 = EncoderBlock(depth * 2, depth * 4, use_dropout=False)
        self.s3_conv4 = EncoderBlock(depth * 4, depth * 8, use_dropout=False)
        self.s3_conv5 = EncoderBlock(depth * 8, depth * 8)
        self.s3_conv6 = EncoderBlock(depth * 8, depth * 8)

        self.deconv1 = DecoderBlock(depth * 24, depth * 8)
        self.deconv2 = DecoderBlock(depth * 32, depth * 8)
        self.deconv3 = DecoderBlock(depth * 32, depth * 4)
        self.deconv4 = DecoderBlock(depth * 16, depth * 2)
        self.deconv5 = DecoderBlock(depth * 8, depth)
        self.deconv6 = DecoderBlock(depth * 4, img_channels, use_dropout=False)

        self.transfer = WrapperNetwork(config)

        self.combo = nn.Conv2d(2 * img_channels, img_channels, kernel_size, stride=1, padding=1)

    def forward(self, landsat, sentinel, cloudy, ground_truth, mask, cluster):
        rem = self.shortcut(cloudy)
        landsat, cloudy, sentinel = landsat.double(), cloudy.double(), sentinel.double()
        landsat = self.upsample(landsat)

        # one-hot encoding clusters
        cluster = F.one_hot(cluster.to(torch.int64)).double()
        cluster = cluster.view(-1, 4, self.img_shape[1], self.img_shape[2])

        # concat mask
        mask_c = mask[:, 0:1, :, :]
        #         mask_d = mask[:, 0:1, ::self.factor, ::self.factor]
        if self.config['use_mask']:
            cloudy_m = torch.cat((cloudy, mask_c), 1)
            sentinel_m = torch.cat((sentinel, mask_c), 1)
            landsat_m = torch.cat((landsat, mask_c), 1)

        cloudy1 = self.s2_conv1(cloudy_m)
        cloudy2 = self.s2_conv2(cloudy1)
        cloudy3 = self.s2_conv3(cloudy2)
        cloudy4 = self.s2_conv4(cloudy3)
        cloudy5 = self.s2_conv5(cloudy4)
        cloudy6 = self.s2_conv6(cloudy5)

        landsat1 = self.s1_conv1(landsat_m)
        landsat2 = self.s1_conv2(landsat1)
        landsat3 = self.s1_conv3(landsat2)
        landsat4 = self.s1_conv4(landsat3)
        landsat5 = self.s1_conv5(landsat4)
        landsat6 = self.s1_conv6(landsat5)

        sentinel1 = self.s3_conv1(sentinel_m)
        sentinel2 = self.s3_conv2(sentinel1)
        sentinel3 = self.s3_conv3(sentinel2)
        sentinel4 = self.s3_conv4(sentinel3)
        sentinel5 = self.s3_conv5(sentinel4)
        sentinel6 = self.s3_conv6(sentinel5)

        deconv1 = self.deconv1(None, landsat6, cloudy6, sentinel6)
        deconv2 = self.deconv2(deconv1, landsat5, cloudy5, sentinel5)
        deconv3 = self.deconv3(deconv2, landsat4, cloudy4, sentinel4)
        deconv4 = self.deconv4(deconv3, landsat3, cloudy3, sentinel3)
        deconv5 = self.deconv5(deconv4, landsat2, cloudy2, sentinel2)
        x = self.deconv6(deconv5, landsat1, cloudy1, sentinel1)

        stylized_global = ada_in(x, cloudy, mask)

        #         x += rem

        mse_loss = nn.MSELoss()
        ground_truth = ground_truth.double()
        # eps = torch.Tensor([1e-5]).double().cuda()
        # print(x.dtype)

        # if self.training:
        #     loss = mse_loss(stylized_global, ground_truth)
        #     return stylized_global, loss

        style = stylized_global * mask + cloudy * (1. - mask)
        x_all = self.transfer(style, cluster, mask)

        # stylized_patch = conv_ada_in_weighted(x, style, mask, patch_size=12, stride=12, dilate_rate=1, padding=0)
        # stylized_patch = ada_in_cluster(x, style, mask, cluster)
        # patch_global = torch.cat((stylized_patch, stylized_global), 1)
        # stylized = self.combo(patch_global)
        # stylized = torch.nan_to_num(stylized)

        # pred = ada_in(x_all, cloudy, mask)
        pred = torch.cat((x_all, style), 1)
        pred = self.combo(pred)
        pred = pred * mask + cloudy * (1. - mask)
        loss = mse_loss(pred, ground_truth)
        return pred, loss, loss


class WrapperNetwork(nn.Module):
    def __init__(self, config):
        super(WrapperNetwork, self).__init__()
        self.config = config
        in_channels, out_channels = self.config['in_channels'], self.config['out_channels']
        feat = self.config['feat']
        activation = self.config['activation']
        kernel_size = self.config['kernel_size']
        num_layers = self.config['num_layers']
        img_channels = self.config['img_channels']
        self.factor = self.config['downsample_factor']
        depth = self.config['depth']
        self.batch_size = self.config['batch_size']
        self.img_shape = self.config["img_shape"]

        self.conv1 = EncoderBlock(img_channels, depth, norm=False, use_dropout=False)
        self.conv2 = EncoderBlock(depth, depth * 2, use_dropout=False)
        self.conv3 = EncoderBlock(depth * 4, depth * 8)
        self.conv4 = EncoderBlock(depth * 8, depth * 8)

        self.pool1 = nn.AvgPool2d(2, padding=0)
        self.pool2 = nn.AvgPool2d(2, padding=0)
        self.padain = PatchTransferCluster(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True)

        self.combo = nn.Conv2d(depth * 4, depth * 4, kernel_size, padding_mode='zeros', padding=1)

        self.deconv1 = DecoderBlock(depth * 8, depth * 8)
        self.deconv2 = DecoderBlock(depth * 16, depth * 2)
        self.deconv3 = DecoderBlock(depth * 4, depth)
        self.deconv4 = DecoderBlock(depth * 2, img_channels, use_dropout=False)

    def forward(self, xin, cluster, mask):
        cluster = self.pool1(cluster)
        cluster = self.pool2(cluster)

        mask_d = self.pool1(mask)
        mask_d = self.pool2(mask_d)

        encode1 = self.conv1(xin)
        encode2 = self.conv2(encode1)
        x_pm = self.padain(cluster, encode2, mask_d)

        pm_encode2 = torch.cat((encode2, x_pm), 1)
        pm_encode2 = self.combo(pm_encode2)

        encode3 = self.conv3(pm_encode2)
        encode4 = self.conv4(encode3)

        decode1 = self.deconv1(None, encode4, None, None)
        decode2 = self.deconv2(decode1, encode3, None, None)
        decode3 = self.deconv3(decode2, encode2, None, None)
        decode4 = self.deconv4(decode3, encode1, None, None)

        return decode4


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]