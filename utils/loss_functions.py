import torch
import torchprofile
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
import sys
sys.path.insert(0, "..")
from utils.iqa import SSIM

class LossFunctions:
    def __init__(self):
        self.l1_loss_fn = F.l1_loss
        self.ssim_loss_fn = SSIM
        self.freq_amp_loss_fn = self.frequency_amplitude_loss
        self.freq_phase_loss_fn = self.frequency_phase_loss
        self.color_loss_fn = self.chroma_contrast_loss

    # Perceptual
    def frequency_amplitude_loss(self, pred, target):
        # Ensure both inputs are complex-compatible tensors
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # Compute amplitude spectrum
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        # Use normalized L1 difference for stability
        return F.l1_loss(pred_amp, target_amp)
    
    def frequency_phase_loss(self, pred, target):
        # Ensure both inputs are complex-compatible tensors
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # Compute phase spectra
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)

        # Use normalized L1 difference for stability
        return F.l1_loss(pred_phase, target_phase)

    def chroma_contrast_loss(self, pred, target):
        pred_lab = self.rgb_to_lab((pred + 1) / 2)
        target_lab = self.rgb_to_lab((target + 1) / 2)

        # get a,b channels
        pa, pb = pred_lab[:,1], pred_lab[:,2]
        ta, tb = target_lab[:,1], target_lab[:,2]

        # chroma maps
        C1 = torch.sqrt(pa*pa + pb*pb + 1e-6)
        C2 = torch.sqrt(ta*ta + tb*tb + 1e-6)

        # **local contrast via Sobel**
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)

        C1_grad = torch.abs(F.conv2d(C1.unsqueeze(1), sobel_x, padding=1)) + \
                  torch.abs(F.conv2d(C1.unsqueeze(1), sobel_y, padding=1))

        C2_grad = torch.abs(F.conv2d(C2.unsqueeze(1), sobel_x, padding=1)) + \
                  torch.abs(F.conv2d(C2.unsqueeze(1), sobel_y, padding=1))

        return torch.mean(torch.abs(C1_grad - C2_grad))
    
    def rgb_to_lab(self, img):
        # sRGB → Linear RGB
        mask = (img > 0.04045)
        img_lin = torch.where(
            mask,
            ((img + 0.055) / 1.055) ** 2.4,
            img / 12.92
        )

        # Linear RGB → XYZ
        rgb_to_xyz = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], dtype=img.dtype, device=img.device)

        # convert B×3×H×W → B×H×W×3 for matmul
        img_lin = img_lin.permute(0, 2, 3, 1)

        # per-pixel matrix multiplication
        xyz = torch.einsum("bhwc,cd -> bhwd", img_lin, rgb_to_xyz)

        # Normalize for D65
        xyz_white = torch.tensor(
            [0.950456, 1.0, 1.088754],
            dtype=img.dtype, device=img.device
        )
        xyz = xyz / xyz_white

        # XYZ → f(XYZ)
        eps = 6/29
        thr = eps ** 3

        xyz_f = torch.where(
            xyz > thr,
            xyz ** (1/3),
            (xyz / (3 * eps**2)) + (4/29)
        )

        # f(X), f(Y), f(Z) → LAB
        fX, fY, fZ = xyz_f[..., 0], xyz_f[..., 1], xyz_f[..., 2]

        L = 116 * fY - 16
        a = 500 * (fX - fY)
        b = 200 * (fY - fZ)

        # stack and permute to B×3×H×W
        lab = torch.stack([L, a, b], dim=-1).permute(0, 3, 1, 2)

        return lab