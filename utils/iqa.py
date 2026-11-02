import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ms_ssim
import lpips

from torchmetrics.image.fid import FrechetInceptionDistance

from pyiqa import create_metric

import numpy as np
from scipy.stats import kurtosis

# PSNR algorithm
class PSNR(nn.Module):
    def __init__(self, data_range=2.0): # if inputs in [-1,1], data_range=2
        super(PSNR, self).__init__()
        self.data_range = data_range

    def forward(self, pred, gt):
        mse = F.mse_loss(pred, gt, reduction='none').mean(dim=[1, 2, 3])
        psnr_batch = 10.0 * torch.log10((self.data_range ** 2) / (mse + 1e-8))
        return psnr_batch.mean()
'''
# PSNR calculation
def psnr(pred, gt, data_range=2.0): # if inputs in [-1,1], data_range=2
    mse = F.mse_loss(pred, gt, reduction='none')
    mse = mse.mean(dim=[1,2,3])
    psnr_batch = 10.0 * torch.log10((data_range ** 2) / (mse + 1e-8))
    return psnr_batch.mean().item()
'''

# SSIM algorithm
SSIM = ms_ssim

# LPIPS algorithm
LPIPS = lpips.LPIPS(net='alex')

# BRISQUE algorithm
BRISQUE = create_metric('brisque', device='cpu')

# NIQE algorithm
NIQE = create_metric('niqe', device='cpu')

# PIQE algorithm
PIQE = create_metric('piqe', device='cpu')

# FID algorithm
FID = FrechetInceptionDistance(feature=2048, normalize=True)

# FADE (Fog Aware Density Evaluator) algorithm
# https://doi.org/10.1109/TIP.2015.2456502
class FADE(nn.Module):
    def __init__(self, omega=0.95, block_size=15):
        super(FADE, self).__init__()
        self.omega = omega
        self.block_size = block_size

    def forward(self, image):
        # Calculate dark channel
        pad_size = self.block_size // 2
        min_channel = -F.max_pool2d(-image, kernel_size=self.block_size, stride=1, padding=pad_size)
        dark_channel = torch.min(min_channel, dim=1, keepdim=True)[0]

        # Calculate atmospheric light
        batch_size, _, H, W = image.shape
        num_pixels = H * W
        num_top_pixels = int(max(num_pixels * 0.001, 1))
        dark_flat = dark_channel.view(batch_size, -1)
        _, indices = torch.topk(dark_flat, num_top_pixels, largest=True)
        A = []
        for i in range(batch_size):
            A.append(torch.max(image[i].view(3, -1)[:, indices[i]], dim=1)[0].mean())
        atmospheric_light = torch.tensor(A, device=image.device).view(batch_size, 1, 1, 1).repeat(1, 3, 1, 1)

        # Calculate transmission map
        transmission_map = 1 - self.omega * dark_channel / (atmospheric_light + 1e-8)
        transmission_map = torch.clamp(transmission_map, min=0.1, max=1.0)

        # Calculate FADE score
        fade_score = torch.mean((image - atmospheric_light) ** 2 / (transmission_map + 1e-8), dim=(1, 2, 3))

        return fade_score