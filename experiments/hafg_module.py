import torch
import torch.nn as nn
import torch.nn.functional as F

def clamp_with_grad(x, min=1e-10, max=1e10):
    return x.detach().clamp(min, max) + (x - x.detach())

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# Prior-guided skip connections (PGM): haze-aware guidance G' for the skip connections
class PGM(nn.Module):
    def __init__(self, omega=0.90, omega_learnable=True, window_size=7, haze_p=0.1,
                 guidance_channels=32, skip_in_channels=(32, 64), pool='min'):
        super().__init__()
        assert pool in ('min', 'max', 'mean'), "pool must be min, max or mean"
        self.window_size, self.haze_p, self.pool = window_size, haze_p, pool
        self.omega = nn.Parameter(torch.tensor(float(omega)), requires_grad=omega_learnable)
        # One mapping phi per encoder level, keyed by that level's channel count
        self.phi = nn.ModuleDict({
            str(c): nn.Sequential(
                nn.Conv2d(2 * c, guidance_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(guidance_channels, guidance_channels, kernel_size=1)
            ) for c in skip_in_channels})

    def haze_density(self, x01):
        """D: minimum (or max / mean) over channels, then over a local patch. (B, C, H, W) -> (B, 1, H, W)"""
        padding = self.window_size // 2
        def patch(fn, z):
            return fn(F.pad(z, (padding, padding, padding, padding), mode='reflect'), kernel_size=self.window_size, stride=1)
        if self.pool == 'max':
            return patch(F.max_pool2d, x01.max(dim=1, keepdim=True).values)
        if self.pool == 'mean':
            return patch(F.avg_pool2d, x01.mean(dim=1, keepdim=True))
        return -patch(F.max_pool2d, -x01.min(dim=1, keepdim=True).values)

    def to_skip(self, feat):
        key = str(int(feat.shape[1]))  # int(): under jit tracing shape entries are tensors
        x01 = (feat + 1) / 2
        D = self.haze_density(x01)
        # Haze strength: mean of the lowest p% values of D, per image
        flat = D.flatten(1)
        k = max(1, int(flat.shape[1] * self.haze_p))
        haze_strength = flat.topk(k, dim=1, largest=False).values.mean(dim=1).view(-1, 1, 1, 1)
        t = 1 - self.omega * D
        J = t * x01 + (1 - t) * haze_strength
        return self.phi[key](torch.cat([feat, J], dim=1))

# Frequency Adaptive Feature Modulation
class FAFM(nn.Module):
    def __init__(self, channels, texture_threshold=0.05):
        super().__init__()
        self.amp_conv = ConvBlock(channels, channels)
        self.phase_conv = ConvBlock(channels, channels)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        nn.init.kaiming_normal_(self.fuse.weight, nonlinearity='relu')
        nn.init.zeros_(self.fuse.bias)
        self.texture_threshold = texture_threshold

    def forward(self, x):
        mean = F.avg_pool2d(x, 3, stride=1, padding=1)
        sq_mean = F.avg_pool2d(x**2, 3, stride=1, padding=1)
        std = torch.sqrt(clamp_with_grad(sq_mean - mean**2, min=1e-6))
        texture_mask = (std > self.texture_threshold).float()

        fft = torch.fft.fft2(x)
        amp = torch.sqrt(fft.real**2 + fft.imag**2)
        phase = torch.atan2(fft.imag, fft.real)
        x_amp = torch.fft.ifft2(torch.polar(amp + self.amp_conv(amp), phase + self.phase_conv(phase))).real
        x_phase = torch.fft.ifft2(torch.polar(amp, phase)).real
        x_fuse = self.fuse(torch.cat([x_amp, x_phase], dim=1))
        return texture_mask * x_fuse + (1 - texture_mask) * x

# Spatial Attention Feature Reconstruction
class SAFR(nn.Module):
    def __init__(self, channels, K=8):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.K = max(1, min(K, channels))

    def forward(self, x):
        B, C, H, W = x.shape

        scores = torch.sigmoid(self.norm(x))
        ch_score = scores.mean(dim=[2,3])

        idx = torch.argsort(ch_score, dim=1)

        groups = idx.chunk(self.K, dim=1)

        group_features = []
        for g in groups:
            g_x = x.gather(1, g[:, :, None, None].expand(-1, -1, H, W))
            group_features.append(g_x)

        mixed = []
        for i in range(self.K):
            g1 = group_features[i]
            g2 = group_features[(i + 1) % self.K]
            mixed.append(0.7 * g1 + 0.3 * g2)

        return torch.cat(mixed, dim=1)

# Soft Masked Adaptive Refinement Filter
class SMARF(nn.Module):
    def __init__(self, channels, mid_channels=92, kernels=(3, 5, 7)):
        super().__init__()
        self.kernels = kernels
        # Each branch: kxk convolution to mid_channels, then a 1x1 convolution back to the input channels
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, mid_channels, k, padding=k // 2),
                          nn.Conv2d(mid_channels, channels, 1)) for k in kernels])
        self.mask_logits = nn.Conv2d(channels, len(kernels), kernel_size=1, bias=False)
        for branch in self.branches:
            nn.init.kaiming_normal_(branch[0].weight, nonlinearity='relu')
            nn.init.zeros_(branch[0].bias)

    def forward(self, x):
        masks = torch.softmax(self.mask_logits(x), dim=1)
        out = x
        for i, branch in enumerate(self.branches):
            out = out + masks[:, i:i + 1] * branch(x)
        return out

# Chroma Contrast Feature Module
class CCFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_x.transpose(2, 3).contiguous())
        self.chroma_conv = nn.Conv2d(2, channels, 3, padding=1)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, feat, image):
        rgb = (image + 1) / 2
        cb = -0.168736 * rgb[:, 0:1] - 0.331264 * rgb[:, 1:2] + 0.5 * rgb[:, 2:3]
        cr = 0.5 * rgb[:, 0:1] - 0.418688 * rgb[:, 1:2] - 0.081312 * rgb[:, 2:3]
        chroma = torch.cat([cb, cr], dim=1)
        padded = F.pad(chroma, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(padded, self.sobel_x.repeat(2, 1, 1, 1), groups=2)
        gy = F.conv2d(padded, self.sobel_y.repeat(2, 1, 1, 1), groups=2)
        grad = F.adaptive_avg_pool2d(torch.sqrt(gx * gx + gy * gy + 1e-6), feat.shape[-2:])
        return torch.softmax(self.fuse(torch.cat([feat, self.chroma_conv(grad)], dim=1)), dim=1)

# Hybrid Feature Generator: gated fusion of the FAFM, SAFR, SMARF and CCFM branches
class HFG(nn.Module):
    def __init__(self, in_channels, use_fafm=True, use_safr=True, use_smarf=True, use_chroma=True, smarf_mid_channels=92):
        super().__init__()
        self.use_chroma = use_chroma
        self.branches = nn.ModuleDict()
        if use_fafm:
            self.branches['fafm'] = FAFM(in_channels)
        if use_safr:
            self.branches['safr'] = SAFR(in_channels)
        if use_smarf:
            self.branches['smarf'] = SMARF(in_channels, smarf_mid_channels)
        if use_chroma:
            self.branches['ccfm'] = CCFM(in_channels)
        self.gate = nn.Conv2d(in_channels, max(1, len(self.branches)), 1)

    def forward(self, x, image):
        if len(self.branches) == 0:
            return x
        outputs = [branch(x, image) if name == 'ccfm' else branch(x) for name, branch in self.branches.items()]
        weights = torch.softmax(self.gate(x), dim=1)
        return sum(weights[:, i:i + 1] * out for i, out in enumerate(outputs))


class DehazeModel(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, depth=2,
                 use_pgm=True, use_fafm=True, use_safr=True, use_smarf=True, use_chroma=True,
                 omega=0.90, omega_learnable=True,
                 smarf_mid_channels=92, guidance_channels=32, pgm_window=7, pgm_p=0.1, pgm_pool='min'):
        super().__init__()
        self.depth = depth
        self.base_channels = base_channels
        self.guidance_channels = guidance_channels
        self.use_pgm = use_pgm

        enc_channels = [base_channels * (2 ** i) for i in range(depth)]
        self.enc_channels = enc_channels

        if use_pgm:
            # Attribute name kept as 'physical' so that the checkpoint keys stay 'physical.*'
            self.physical = PGM(omega=omega, omega_learnable=omega_learnable,
                                guidance_channels=guidance_channels, skip_in_channels=enc_channels,
                                window_size=pgm_window, haze_p=pgm_p, pool=pgm_pool)

        self.enc_blocks = nn.ModuleList()
        for i, ch in enumerate(enc_channels):
            self.enc_blocks.append(ConvBlock(in_channels if i == 0 else enc_channels[i - 1], ch))
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = HFG(enc_channels[-1], use_fafm=use_fafm, use_safr=use_safr, use_smarf=use_smarf,
                              use_chroma=use_chroma, smarf_mid_channels=smarf_mid_channels)

        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            prev_dec_ch = enc_channels[-1] if i == 0 else enc_channels[-i]
            enc_ch = enc_channels[-(i + 1)]
            skip_ch = enc_ch + (guidance_channels if use_pgm else 0)
            self.up_blocks.append(UpBlock(prev_dec_ch, skip_ch, enc_ch))

        self.final_conv = nn.Conv2d(enc_channels[0], in_channels, 3, padding=1)

    def forward(self, x):
        feats, skips_guidance = [], []
        cur = x
        for enc in self.enc_blocks:
            cur = clamp_with_grad(enc(cur), min=-5.0, max=5.0)
            feats.append(cur)
            skips_guidance.append(self.physical.to_skip(cur) if self.use_pgm else None)
            cur = self.pool(cur)

        dec = self.bottleneck(cur, x)

        for i, up in enumerate(self.up_blocks):
            enc_idx = len(feats) - 1 - i
            skip = torch.cat([feats[enc_idx], skips_guidance[enc_idx]], dim=1) if self.use_pgm else feats[enc_idx]
            dec = up(dec, skip)

        # The final 3x3 convolution predicts a correction that is added to the input image;
        # the output is clamped to the image range [-1, 1] at inference only
        out = self.final_conv(dec) + x
        return out if self.training else out.clamp(-1.0, 1.0)
