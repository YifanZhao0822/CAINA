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
    # content_mean, content_std = mean_std(content, 1. - mask)
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


def ada_in_cluster(content, style, mask, cluster, patch_size, stride, dilate_rate, padding, softmax_scale=10):
    # get shapes
    content_size = list(content.size())  # b*c*h*w
    style_size = list(style.size())  # b*c*h*w

    upsample = nn.Upsample(scale_factor=stride, mode='nearest')

    # extract patches from background with stride and rate
    #     kernel = 2 * self.rate
    w = extract_image_patches(style, ksizes=[patch_size, patch_size],
                              strides=[stride, stride],
                              rates=[dilate_rate, dilate_rate],
                              padding='same')  # [N, C*k*k, L]
    # raw_shape: [N, C, k, k, L]
    w = w.view(style_size[0], style_size[1], patch_size, patch_size, -1)
    w = w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
    w_groups = torch.split(w, 1, dim=0)

    c = extract_image_patches(content, ksizes=[patch_size, patch_size],
                              strides=[stride, stride],
                              rates=[dilate_rate, dilate_rate],
                              padding='same')  # [N, C*k*k, L]
    # raw_shape: [N, C, k, k, L]
    c = c.view(content_size[0], content_size[1], patch_size, patch_size, -1)
    c = c.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
    c_groups = torch.split(c, 1, dim=0)

    mask_size = list(mask.size())
    # m shape: [N, C*k*k, L]
    m = extract_image_patches(mask, ksizes=[patch_size, patch_size],
                              strides=[stride, stride],
                              rates=[dilate_rate, dilate_rate],
                              padding='same')
    # m shape: [N, C, k, k, L]
    m = m.view(mask_size[0], mask_size[1], patch_size, patch_size, -1)
    m = m.permute(0, 4, 1, 2, 3)  # m shape: [N, L, C, k, k]
    m_groups = torch.split(m, 1, dim=0)

    cluster_size = list(cluster.size())
    cl = extract_image_patches(cluster, ksizes=[patch_size, patch_size],
                              strides=[stride, stride],
                              rates=[dilate_rate, dilate_rate],
                              padding='same')
    # cl shape: [N, k, k, L]
    cl = cl.view(cluster_size[0], cluster_size[1], patch_size, patch_size, -1)
    cl = cl.permute(0, 3, 1, 2)  # m shape: [N, L, k, k]
    cl_groups = torch.split(cl, 1, dim=0)

    y = []
    # iterate over batch
    for xi, wi, mi, cli in zip(c_groups, w_groups, m_groups, cl_groups):
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
            yc = F.softmax(yc * softmax_scale, dim=1)
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

    # content_mean, content_std = mean_std(content, 1. - mask)
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


class PatchTransferCluster(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=True):
        super(PatchTransferCluster, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())  # b*c*h*w
        raw_int_bs = list(b.size())  # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate * self.stride,
                                               self.rate * self.stride],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1. / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / self.rate, mode='nearest')
        int_fs = list(f.size())  # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(f, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_fs[0], int_fs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            #             mask = F.interpolate(mask, scale_factor=1./(4*self.rate), mode='nearest')
            mask = F.interpolate(mask, scale_factor=1. / (1 * self.rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [N, L, C, k, k]

        y = []
        k = self.fuse_k
        scale = self.softmax_scale  # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k).double()  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi, mm in zip(f_groups, w_groups, raw_w_groups, m):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''

            # m = m[0]    # m shape: [L, C, k, k]
            # mm shape: [L, 1, 1, 1]
            # print("shapes",mm.shape, m.shape)
            # mm = mm[0]
            mm = (reduce_mean(mm, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
            mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, (int_bs[2] // self.stride) * (int_bs[3] // self.stride), int_fs[2] * int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, (int_bs[2] // self.stride), (int_bs[3] // self.stride), int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, (int_bs[2] // self.stride) * (int_bs[3] // self.stride), int_fs[2] * int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, (int_bs[2] // self.stride), (int_bs[3] // self.stride), int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, (int_bs[2] // self.stride) * (int_bs[3] // self.stride), int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=(wi_center.shape[2] - self.rate) // 2) # / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_bs)

        return y
