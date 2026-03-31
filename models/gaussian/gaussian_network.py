
import torch
from torch import nn
from .extractor import UnetExtractor, ResidualBlock
from einops import rearrange
import torch.nn.functional as F


class GaussianParamHead(nn.Module):
    """
    单尺度高斯参数预测头
    
    输入：某尺度的特征图 [N, C, H, W]
    输出：该尺度的高斯参数字典（空间维度保持）
        - rot: [N, 4, H, W]，归一化四元数
        - scale: [N, 3, H, W]，经过 Softplus 和 clamp
        - opacity: [N, 1, H, W]，Sigmoid 激活
        - sh_map: [N, 3*d_sh, H, W]，球谐系数图（未 reshape）
    """
    def __init__(self, in_channels, sh_degree=4, max_scale=0.05):
        super().__init__()
        self.sh_degree = sh_degree
        self.d_sh = (sh_degree + 1) ** 2
        self.max_scale = max_scale
        
        # 注册 SH mask（与原始代码一致）
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        
        # Rot 头：输出 4 通道四元数
        self.rot_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1),
        )
        
        # Scale 头：输出 3 通道
        self.scale_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )
        
        # Opacity 头：输出 1 通道
        self.opacity_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # SH 头：输出 3 * d_sh 通道（保持空间维度）
        self.sh_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3 * self.d_sh, kernel_size=1),
        )
    
    def forward(self, x):
        # Rot: 归一化四元数
        rot = self.rot_head(x)
        rot = torch.nn.functional.normalize(rot, dim=1)
        
        # Scale: Softplus + clamp
        scale = torch.clamp_max(self.scale_head(x), self.max_scale)
        
        # Opacity: Sigmoid
        opacity = self.opacity_head(x)
        
        # SH: 保持空间维度 [N, 3*d_sh, H, W]
        sh_map = self.sh_head(x)
        # 应用 SH mask（逐通道相乘）
        sh_map = sh_map * self.sh_mask.view(1, self.d_sh, 1, 1).repeat(1, 3, 1, 1)
        
        return {
            'rot': rot,           # [N, 4, H, W]
            'scale': scale,       # [N, 3, H, W]
            'opacity': opacity,   # [N, 1, H, W]
            'sh_map': sh_map,     # [N, 3*d_sh, H, W]
        }


class GaussianNetwork(nn.Module):
    def __init__(self, rgb_dim=3, depth_dim=1, norm_fn='group'):
        """
        Args:
            rgb_dim: RGB 输入通道数
            depth_dim: 深度输入通道数
            norm_fn: 归一化函数类型


        """
        super().__init__()
        self.rgb_dims = [64, 64, 128]
        self.depth_dims = [32, 48, 96]
        self.decoder_dims = [48, 64, 96]
        self.head_dim = 32

        self.sh_degree = 4
        self.d_sh = (self.sh_degree + 1) ** 2

        self.max_gaussian_scale = 0.05

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0]+rgb_dim+1, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        # ============================================
        # Level 1 (finest): 保留原始的单头结构（用于加载预训练权重）
        # ============================================
        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.sh_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3 * self.d_sh, kernel_size=1),
        )
        
        # ============================================
        # Level 2 (middle) 和 Level 3 (coarsest): 使用新的 GaussianParamHead
        # ============================================
        # Level 2: 中等分辨率，中等尺度
        # 分辨率：[N, head_dim, H/2, W/2]
        self.gaussian_head_level_2 = GaussianParamHead(
            in_channels=self.head_dim,
            sh_degree=self.sh_degree,
            max_scale=self.max_gaussian_scale * 4.0
        )
        
        # Level 3: 最低分辨率，最大尺度
        # 分辨率：[N, head_dim, H/4, W/4]
        self.gaussian_head_level_3 = GaussianParamHead(
            in_channels=self.head_dim,
            sh_degree=self.sh_degree,
            max_scale=self.max_gaussian_scale * 16.0
        )
        

    def forward(self, img, depth, img_feat, xyz, valid, return_debug_masks=False):
        """
        前向传播
        
        Args:
            img: RGB 图像 [N, 3, H, W]
            depth: 深度图 [N, 1, H, W]
            img_feat: 图像特征元组 (feat1, feat2, feat3)
                - feat1: [N, 64, H, W]
                - feat2: [N, 64, H/2, W/2]
                - feat3: [N, 128, H/4, W/4]
            xyz: Level 1 的 3D 点位置图 [N, 3, H, W]（必须输入）
            valid: Level 1 的有效性 mask [N, 1, H, W]（可选）
            return_debug_masks: 是否返回调试 mask（用于可视化）
        
        Returns:
            final_params: 最终高斯参数字典
                {
                    'rot': [N, total_G, 4],
                    'scale': [N, total_G, 3],
                    'opacity': [N, total_G, 1],
                    'sh': [N, total_G, 1, 1, 3, d_sh],
                    'xyz': [N, total_G, 3],  # 高斯中心位置
                    'valid': [N, total_G, 1]  # 有效性标记（如果有输入）
                }
            debug_info (optional): 调试信息字典
        """
        # 分辨率说明：
        # Level 1 (finest): [N, C, H, W]
        # Level 2 (middle): [N, C, H/2, W/2]
        # Level 3 (coarsest): [N, C, H/4, W/4]
        
        # img_feat1: [N, 64, H, W]
        # img_feat2: [N, 64, H/2, W/2]
        # img_feat3: [N, 128, H/4, W/4]
        img_feat1, img_feat2, img_feat3 = img_feat
        
        # depth_feat1: [N, 32, H, W]
        # depth_feat2: [N, 48, H/2, W/2]
        # depth_feat3: [N, 96, H/4, W/4]
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)

        # Decoder 前向传播
        up3 = self.decoder3(feat3)  # [N, 96, H/4, W/4]
        up3 = self.up(up3)  # [N, 96, H/2, W/2]
        
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))  # [N, 64, H/2, W/2]
        up2 = self.up(up2)  # [N, 64, H, W]
        
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))  # [N, 48, H, W]

        up1 = self.up(up1)  # [N, 48, 2H, 2W]
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)  # [N, 32, H, W]
        out = self.out_relu(out)
        
        # 生成三个层级的特征图
        # Level 1: 原始分辨率 [N, 32, H, W]
        out_level_1 = out
        
        # Level 2: 下采样到一半分辨率 [N, 32, H/2, W/2]
        out_level_2 = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # Level 3: 下采样到 1/4 分辨率 [N, 32, H/4, W/4]
        out_level_3 = F.interpolate(out, scale_factor=0.25, mode='bilinear', align_corners=False)

        # ============================================
        # 通过三个 head 预测各层级高斯参数
        # ============================================
        # Level 1: 使用原始的 rot_head/scale_head/opacity_head/sh_head
        rot_out = self.rot_head(out_level_1)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)
        scale_out = torch.clamp_max(self.scale_head(out_level_1), self.max_gaussian_scale)
        opacity_out = self.opacity_head(out_level_1)
        sh_out_map = self.sh_head(out_level_1)
        # 应用 SH mask
        
        sh_out_map = sh_out_map * self.sh_mask.view(1, self.d_sh, 1, 1).repeat(1, 3, 1, 1)

        params_l1 = {
            'rot': rot_out,           # [N, 4, H, W]
            'scale': scale_out,       # [N, 3, H, W]
            'opacity': opacity_out,   # [N, 1, H, W]
            'sh_map': sh_out_map,     # [N, 3*d_sh, H, W]
        }
        
        # Level 2 和 Level 3: 使用新的 GaussianParamHead
        params_l2 = self.gaussian_head_level_2(out_level_2)  # middle
        params_l3 = self.gaussian_head_level_3(out_level_3)  # coarsest

        # ============================================
        # 计算各层级的高斯中心位置（硬性几何对齐约束）
        # ============================================
        # Level 1: 直接使用输入的 3D 点位置
        # xyz: [N, H*W, 3] -> xyz_level_1: [N, 3, H, W]
        xyz_level_1 = xyz.reshape(-1, img.shape[2], img.shape[3], 3).permute(0, 3, 1, 2)
        
        # Level 2: 由 level1 的 2x2 平均池化得到
        # xyz_level_2: [N, 3, H/2, W/2]
        xyz_level_2 = F.avg_pool2d(xyz_level_1, kernel_size=2, stride=2)
        
        # Level 3: 由 level2 的 2x2 平均池化得到
        # xyz_level_3: [N, 3, H/4, W/4]
        xyz_level_3 = F.avg_pool2d(xyz_level_2, kernel_size=2, stride=2)
        
        # 添加到参数中
        params_l1['xyz'] = xyz_level_1
        params_l2['xyz'] = xyz_level_2
        params_l3['xyz'] = xyz_level_3
        
        # ============================================
        # 处理有效性 mask（如果有输入）
        # ============================================
        if valid is not None:
            # Level 1: 直接使用输入的 valid mask
            valid_level_1 = valid.float()  # [N, 1, H, W]
            
            # Level 2: 由 level1 的 2x2 最大池化得到（只要有一个有效就有效）
            valid_level_2 = F.max_pool2d(valid_level_1, kernel_size=2, stride=2)
            
            # Level 3: 由 level2 的 2x2 最大池化得到
            valid_level_3 = F.max_pool2d(valid_level_2, kernel_size=2, stride=2)
            
            # 添加到参数中
            params_l1['valid'] = valid_level_1
            params_l2['valid'] = valid_level_2
            params_l3['valid'] = valid_level_3

        # ============================================
        # 层级梯度分配（基于结构复杂度）
        # ============================================
        # no together
        final_params = {"level_1": params_l1, "level_2": params_l2, "level_3": params_l3}
        for scale in ["level_1", "level_2", "level_3"]:
            final_params[scale]['sh_map'] = rearrange(final_params[scale]['sh_map'], "n (xyz d_sh) h w -> n (h w) 1 1 xyz d_sh", xyz=3)
            final_params[scale]['valid'] = final_params[scale]['valid'].bool()
        return final_params
