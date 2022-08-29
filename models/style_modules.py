from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import extract_image_patches, same_padding, reduce_mean, reduce_sum


def mean_std(feat, mask, eps=1e-5):
    mask = mask <= 0
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_mean, feat_std = torch.zeros(N, C, 1, 1), torch.zeros(N, C, 1, 1)
    for i in range(N):
        for j in range(C):
            masked = torch.masked_select(feat[i, j, :, :], mask[i, j, :, :])
            feat_mean[i, j, 0, 0] = masked.mean()
            feat_std[i, j, 0, 0] = (masked.var() + eps).sqrt()
    
    return feat_mean, feat_std
    
def ada_in(content, style, mask=None):
    # input with nbatch: (N, C, H, W)
    # content=landsat, style=cloudy
    assert (content.size()[:2] == style.size()[:2])
    size = content.size()
    
    if mask == None:
        mask = torch.zeros(size).cuda()
    
    style_mean, style_std = mean_std(style, mask)
    content_mean, content_std = mean_std(content, torch.zeros(size).cuda())
#     print("style_mean, style_std", style_mean.shape, style_std.shape)
#     print("content_mean, content_std", content_mean.shape, content_std.shape)
    
    normalized_feat = (content - content_mean.expand(size).cuda()) / content_std.expand(size).cuda()
#     print("normalized", normalized_feat.shape)
    
    return normalized_feat * style_std.expand(size).cuda() + style_mean.expand(size).cuda()


def comp_ada_in(content, cloudy, sentinel, mask):
    assert (content.size()[:2] == cloudy.size()[:2])
    size = content.size()
    
    style = cloudy + sentinel * mask
    
    style_mean, style_std = mean_std(style, torch.zeros(size).cuda())
    content_mean, content_std = mean_std(content, torch.zeros(size).cuda())
#     print("style_mean, style_std", style_mean.shape, style_std.shape)
#     print("content_mean, content_std", content_mean.shape, content_std.shape)
    
    normalized_feat = (content - content_mean.expand(size).cuda()) / content_std.expand(size).cuda()
#     print("normalized", normalized_feat.shape)
    
    return normalized_feat * style_std.expand(size).cuda() + style_mean.expand(size).cuda()


def ada_in_max(content, style, mask, patch_size, stride, dilate_rate, padding, softmax_scale=10):
    # get shapes
    content_size = list(content.size())   # b*c*h*w
    style_size = list(style.size())   # b*c*h*w

    upsample = nn.Upsample(scale_factor=stride, mode='nearest')

    # extract patches from background with stride and rate
#     kernel = 2 * self.rate
    w = extract_image_patches(style, ksizes=[patch_size, patch_size],
                                  strides=[stride, stride],
                                  rates=[dilate_rate, dilate_rate],
                                  padding='same') # [N, C*k*k, L]
    # raw_shape: [N, C, k, k, L]
    w = w.view(style_size[0], style_size[1], patch_size, patch_size, -1)
    w = w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
    w_groups = torch.split(w, 1, dim=0)

    # no need to downscaling here if we are computing the weights between input and output features
    content_groups = torch.split(content, 1, dim=0)

    mask_size = list(mask.size())
    # m shape: [N, C*k*k, L]
    m = extract_image_patches(mask, ksizes=[patch_size, patch_size],
                                  strides=[stride, stride],
                                  rates=[dilate_rate, dilate_rate],
                                  padding='same')
    # m shape: [N, C, k, k, L]
    m = m.view(mask_size[0], mask_size[1], patch_size, patch_size, -1)
    m = m.permute(0, 4, 1, 2, 3)  # m shape: [N, L, C, k, k]
    mask_groups = torch.split(m, 1, dim=0)
    
    y = []
    # iterate over batch
    for xi, wi, mi in zip(content_groups, w_groups, mask_groups):
        """
        O => output channel as a conv filter
        I => input channel as a conv filter
        xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
        wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=psize, KW=psize)
        """
        escape_NaN = torch.FloatTensor([1e-4]).cuda()

        # iterate over channels
        xi_channels = torch.split(xi, 1, dim=1)
        wi_channels = torch.split(wi, 1, dim=2)
        mi_channels = torch.split(mi, 1, dim=2)
        yi = []
        for xc, wc, mc in zip(xi_channels, wi_channels, mi_channels):

            # conv for compare
            wc = wc[0]  # [L, C, k, k]
            max_wc = torch.sqrt(reduce_sum(torch.pow(wc, 2) + escape_NaN, axis=[1, 2], keepdim=True))
            wc_normed = wc / max_wc
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            # xi = same_padding(xi, [patch_size, patch_size], [1, 1], [1, 1])  # xi: 1*c*H*W

            yc = F.conv2d(xc, wc_normed, stride=stride, dilation=dilate_rate, padding=padding)
            yc = upsample(yc)

            mc = mc[0]
            raw_mc = mc
            mc = reduce_mean(mc, axis=[1, 2, 3], keepdim=True).to(torch.float32)
            # mc = (reduce_sum(mc, axis=[1, 2, 3], keepdim=True) <= patch_size ** 2 / 4.).to(torch.float32)
            mc = mc.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]
            # apply single-channel mask to yc here
            yc = yc.view(1, -1, content_size[2], content_size[3])
            yc = yc * mc
            yc = F.softmax(yc*softmax_scale, dim=1)
            # idx: (1, 1, 384, 384)
            _, idx = torch.max(yc, dim=1, keepdim=True)

            yc = patch_ada_in(xc, wc, idx, raw_mc)
            yi.append(yc)
        # then have the argmax after softmax here and compute the mean and std right here
        yi = torch.cat(yi, dim=1)
        y.append(yi)
    
    y = torch.cat(y, dim=0)  # back to the mini-batch
    y.contiguous().view(content_size)
    return y


def patch_ada_in(content, style_patches, idx, mask, eps=1e-5):
    size = content.size()
    # mean = style_patches.mean(dim=(1, 2, 3))
    # std = (style_patches.var(dim=(1, 2, 3)) + eps).sqrt()
    # with torch.no_grad():
    mean, std = mean_std_patch(style_patches, mask)
    style_mean = torch.index_select(mean.cuda(), 0, idx.view(-1)).view(size).cuda()
    style_std = torch.index_select(std.cuda(), 0, idx.view(-1)).view(size).cuda()

    content_mean, content_std = mean_std(content, torch.zeros(size).cuda())
    normalized_feat = (content - content_mean.expand(size).cuda()) / (content_std + eps).expand(size).cuda()
    normalized_feat = normalized_feat * style_std + style_mean + eps
    # normalized_feat = style_std + style_mean
    return normalized_feat


def mean_std_patch(feat, mask, eps=1e-5):
    mask = mask <= 0
    size = feat.size()
    assert (len(size) == 4)
    L = size[0]
    feat_mean, feat_std = torch.zeros(L, 1, 1, 1), torch.zeros(L, 1, 1, 1)
    for i in range(L):
        masked = torch.masked_select(feat[i, :, :, :], mask[i, :, :, :])
        feat_mean[i, 0, 0, 0] = masked.mean()
        feat_std[i, 0, 0, 0] = (masked.var() + eps).sqrt() + eps

    return feat_mean, feat_std
    

class AdaConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, kernel_size):
        super(AdaConvDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_channels = style_channels
        self.kernel_size = kernel_size
        
        self.kernel_predictor = KernelPredictor(in_channels, in_channels, n_groups=n_groups, style_channels=self.style_channels, kernel_size=self.kernel_size)
        self.ada_conv = AdaConv2d(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect')
        
    def forward(self, content, w_style):
        w_spatial, w_pointwise, bias = self.kernel_predictor(w_style)
        content = self.ada_conv(content, w_spatial, w_pointwise, bias)
        content = self.conv(content)
        return content
    

class StyleEncoder(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(StyleEncoder, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        channels = in_shape[0]

        self.downscale = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
        )

        in_features = self.in_shape[0] * (self.in_shape[1] // 8) * self.in_shape[2] // 8
        out_features = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, xs):
        ys = self.downscale(xs)
        ys = ys.reshape(len(xs), -1)

        w = self.fc(ys)
        w = w.reshape(len(xs), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return w
    
    
class AdaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, kernel_size):
        super().__init__()
        self.style_channels = style_channels
        self.kernel_size = kernel_size
        
        self.kernel_predictor = KernelPredictor(in_channels, in_channels, style_channels=style_channels, kernel_size=kernel_size)
        self.ada_conv = AdaConv2d(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect')
        
    def forward(self, content, w_style):
        w_spatial, w_pointwise, bias = self.kernel_predictor(w_style)
        content = self.ada_conv(content, w_spatial, w_pointwise, bias)
        content = self.conv(content)
        return content
    

class AdaConv2d(nn.Module):
    """
    Implementation of the Adaptive Convolution block. Performs a depthwise seperable adaptive convolution on its input X.
    The weights for the adaptive convolutions are generated by a KernelPredictor module based on the style embedding W.
    The adaptive convolution is followed by a normal convolution.
    References:
        https://openaccess.thecvf.com/content/CVPR2021/papers/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.pdf
    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by final convolution.
        kernel_size: The kernel size of the final convolution.
        n_groups: The number of groups for the adaptive convolutions.
            Defaults to 1 group per channel if None.
    Input shape:
        x: Input tensor.
        w_spatial: Weights for the spatial adaptive convolution.
        w_pointwise: Weights for the pointwise adaptive convolution.
        bias: Bias for the pointwise adaptive convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super().__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(ceil(padding), floor(padding)),
                              padding_mode='reflect')

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)

        # F.conv2d does not work with batched filters (as far as I can tell)...
        # Hack for inputs with > 1 sample
        ys = []
        for i in range(len(x)):
            y = self._forward_single(x[i:i + 1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def _forward_single(self, x, w_spatial, w_pointwise, bias):
        # Only square kernels
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (ceil(padding), floor(padding), ceil(padding), floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x
    
    
class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, kernel_size, n_groups=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_channels = style_channels
        self.n_groups = in_channels if not n_groups else n_groups
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // self.n_groups,
                                 kernel_size=kernel_size,
                                 padding=(ceil(padding), ceil(padding)),
                                 padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // self.n_groups,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(len(w),
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(len(w),
                                          self.out_channels,
                                          self.out_channels // self.n_groups,
                                          1, 1)

        bias = self.bias(w)
        bias = bias.reshape(len(w),
                            self.out_channels)

        return w_spatial, w_pointwise, bias