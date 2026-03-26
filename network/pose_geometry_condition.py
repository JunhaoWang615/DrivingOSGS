import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseRayFieldBuilder(nn.Module):
    """Build ray field: [B, N, 6, H, W] = [center(3), dir(3)]"""
    def __init__(self, image_height=128, image_width=256, eps=1e-6):
        super().__init__()
        self.H = image_height
        self.W = image_width
        self.eps = eps

    def _build_pixel_grid(self, device, dtype):
        # u: [W], v: [H], pixel-center coordinates
        u = torch.arange(self.W, device=device, dtype=dtype) + 0.5
        v = torch.arange(self.H, device=device, dtype=dtype) + 0.5
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # [H, W], [H, W]
        ones = torch.ones_like(uu)
        pix = torch.stack([uu, vv, ones], dim=0)  # [3, H, W]
        return pix

    def forward(self, K, R, t):
        # K: [B, N, 3, 3], R: [B, N, 3, 3], t: [B, N, 3]
        B, N = K.shape[:2]
        device, dtype = K.device, K.dtype

        K_inv = torch.inverse(K)  # [B, N, 3, 3]
        R_t = R.transpose(-1, -2)  # [B, N, 3, 3]

        # optical_center = -R^T @ t
        t_col = t.unsqueeze(-1)  # [B, N, 3, 1]
        center = -torch.matmul(R_t, t_col).squeeze(-1)  # [B, N, 3]

        pix = self._build_pixel_grid(device, dtype).view(1, 1, 3, self.H * self.W)  # [1,1,3,HW]
        pix = pix.expand(B, N, -1, -1)  # [B,N,3,HW]

        bearing = torch.matmul(K_inv, pix)  # [B, N, 3, HW]
        dir_world = torch.matmul(R_t, bearing)  # [B, N, 3, HW]
        dir_world = dir_world / (torch.norm(dir_world, dim=2, keepdim=True) + self.eps)

        dir_world = dir_world.view(B, N, 3, self.H, self.W)  # [B,N,3,H,W]
        center_img = center.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.H, self.W)  # [B,N,3,H,W]

        ray_field = torch.cat([center_img, dir_world], dim=2)  # [B,N,6,H,W]
        return ray_field


class FourierEncoder(nn.Module):
    """NeRF-style Fourier encoding for channel-wise scalar values."""
    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        freq_bands = (2.0 ** torch.arange(num_freqs)) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x):
        # x: [B, N, C, H, W]
        outs = [x]
        for f in self.freq_bands:
            outs.append(torch.sin(f * x))
            outs.append(torch.cos(f * x))
        return torch.cat(outs, dim=2)  # [B,N,C*(1+2L),H,W]


class _GeomStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GeometryEncoder(nn.Module):
    """Encode ray_field [B,N,6,128,256] -> geom_feat [B,N,256,16,32]."""
    def __init__(self, num_freqs=6):
        super().__init__()
        self.fourier = FourierEncoder(num_freqs=num_freqs)
        in_ch = 6 * (1 + 2 * num_freqs)
        self.input_proj = nn.Conv2d(in_ch, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.stage1 = _GeomStage(64, 64)    # 128x256 -> 64x128
        self.stage2 = _GeomStage(64, 128)   # 64x128 -> 32x64
        self.stage3 = _GeomStage(128, 256)  # 32x64 -> 16x32

    def forward(self, ray_field):
        B, N, _, H, W = ray_field.shape
        x = self.fourier(ray_field)  # [B,N,Cf,H,W]
        x = x.reshape(B * N, x.shape[2], H, W)  # merge for Conv2d
        x = self.input_proj(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(B, N, 256, 16, 32)
        return x


class GeometryFeatureFusion(nn.Module):
    """Fuse visual + geometric features at [B,N,256,16,32]."""
    def __init__(self, feat_dim=256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, visual_feat, geom_feat):
        B, N, C, H, W = visual_feat.shape
        x = torch.cat([visual_feat, geom_feat], dim=2)  # [B,N,2C,H,W]
        x = x.reshape(B * N, 2 * C, H, W)
        x = self.fuse(x)
        x = x.view(B, N, C, H, W)
        return x


class PoseGeometricConditionEncoder(nn.Module):
    """Full module: ray_field -> geom_feat -> fused_feat."""
    def __init__(self, feat_dim=256, image_height=128, image_width=256, num_freqs=6):
        super().__init__()
        self.ray_builder = DenseRayFieldBuilder(image_height=image_height, image_width=image_width)
        self.geom_encoder = GeometryEncoder(num_freqs=num_freqs)
        self.fusion = GeometryFeatureFusion(feat_dim=feat_dim)

    def forward(self, visual_feat, K, R, t):
        ray_field = self.ray_builder(K=K, R=R, t=t)       # [B,N,6,128,256]
        geom_feat = self.geom_encoder(ray_field)          # [B,N,256,16,32]
        fused_feat = self.fusion(visual_feat, geom_feat)  # [B,N,256,16,32]
        return fused_feat, geom_feat, ray_field
