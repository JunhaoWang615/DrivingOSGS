
import torch
from torch import nn
from .extractor import UnetExtractor, ResidualBlock
from einops import rearrange
import torch.nn.functional as F
from .hierarchical_gaussian_selector import HierarchicalGaussianSelector


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
    def __init__(self, rgb_dim=3, depth_dim=1, norm_fn='group', tau = 0):
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
            max_scale=self.max_gaussian_scale * 6
        )
        
        # Level 3: 最低分辨率，最大尺度
        # 分辨率：[N, head_dim, H/4, W/4]
        self.gaussian_head_level_3 = GaussianParamHead(
            in_channels=self.head_dim,
            sh_degree=self.sh_degree,
            max_scale=self.max_gaussian_scale * 16
        )

        # Post-processing selector: no learnable parameters.
        self.hierarchical_selector = HierarchicalGaussianSelector(
            lambda_i=0.5,
            lambda_d=0.5,
            tau21=tau,
            tau32=tau,
            eps=1e-6,
        )
        self.last_selector_results = None

    @staticmethod
    def _flatten_spatial_tensor(x):
        # [N, C, H, W] -> [N, H*W, C]
        n, c, h, w = x.shape
        return x.reshape(n, c, h * w).transpose(1, 2).contiguous()

    def _build_level_multi_from_masks(self, img, params_l1, params_l2, params_l3, masks):
        def _select_by_mask(param_dict, mask):
            # mask: [N, 1, H, W] (bool)
            selected = {}
            flat_mask = mask.reshape(mask.shape[0], -1)
            for k, v in param_dict.items():
                if not torch.is_tensor(v):
                    continue
                if v.dim() == 4 and v.shape[-2:] == mask.shape[-2:]:
                    v_flat = self._flatten_spatial_tensor(v)
                    selected[k] = [v_flat[b][flat_mask[b]] for b in range(v.shape[0])]
            return selected

        sel_l1 = _select_by_mask(params_l1, masks['m1'])
        sel_l2 = _select_by_mask(params_l2, masks['m2'])
        sel_l3 = _select_by_mask(params_l3, masks['m3'])

        merge_keys = ['rot', 'scale', 'opacity', 'sh_map', 'xyz']
        feat_dims = {
            'rot': 4,
            'scale': 3,
            'opacity': 1,
            'sh_map': 3 * self.d_sh,
            'xyz': 3,
        }

        per_batch_concat = {k: [] for k in merge_keys}
        batch_size = img.shape[0]
        for b in range(batch_size):
            for k in merge_keys:
                parts = []
                if k in sel_l1:
                    parts.append(sel_l1[k][b])
                if k in sel_l2:
                    parts.append(sel_l2[k][b])
                if k in sel_l3:
                    parts.append(sel_l3[k][b])

                if len(parts) == 0:
                    empty = img.new_zeros((0, feat_dims[k]))
                    per_batch_concat[k].append(empty)
                else:
                    per_batch_concat[k].append(torch.cat(parts, dim=0))

        max_count = 0
        for b in range(batch_size):
            max_count = max(max_count, per_batch_concat['rot'][b].shape[0])

        level_multi = {}
        for k in merge_keys:
            feat_dim = feat_dims[k]
            stacked = img.new_zeros((batch_size, max_count, feat_dim))
            for b in range(batch_size):
                cnt = per_batch_concat[k][b].shape[0]
                if cnt > 0:
                    stacked[b, :cnt] = per_batch_concat[k][b]
            level_multi[k] = stacked

        level_multi['valid'] = torch.zeros(
            (batch_size, max_count, 1),
            dtype=torch.bool,
            device=img.device,
        )
        for b in range(batch_size):
            cnt = per_batch_concat['rot'][b].shape[0]
            if cnt > 0:
                level_multi['valid'][b, :cnt, 0] = True

        level_multi['sh_map'] = rearrange(
            level_multi['sh_map'],
            'n g (xyz d_sh) -> n g 1 1 xyz d_sh',
            xyz=3,
            d_sh=self.d_sh,
        )
        return level_multi
        

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

        # Hierarchical hard selection (mask-based selection + bookkeeping).
        selector_results = self.hierarchical_selector(
            image=img,
            depth=depth,
            gaussians_l1=params_l1,
            gaussians_l2=params_l2,
            gaussians_l3=params_l3,
            valid_mask=params_l1['valid'] if 'valid' in params_l1 else None,
        )
        self.last_selector_results = selector_results
        masks = selector_results['aggregation_masks']
        level_multi = self._build_level_multi_from_masks(
            img=img,
            params_l1=params_l1,
            params_l2=params_l2,
            params_l3=params_l3,
            masks=masks,
        )
# import math; xx =torch.log(1.0 + selector_results['complexity']['c1'] * (math.e - 1.0)) ** 0.8; import torch; import torchvision.utils as vutils; vutils.save_image(xx, "test_img_block/complexity.png", normalize=True)
        # ============================================
        # 层级梯度分配（基于结构复杂度）
        # ============================================
        # no together
        final_params = {
            "level_1": params_l1,
            "level_2": params_l2,
            "level_3": params_l3,
            "level_multi": level_multi,
            "masks": masks,  # 包含 m1, m2, m3
        }
        for scale in ["level_1", "level_2", "level_3"]:
            final_params[scale]['sh_map'] = rearrange(final_params[scale]['sh_map'], "n (xyz d_sh) h w -> n (h w) 1 1 xyz d_sh", xyz=3)
            final_params[scale]['valid'] = final_params[scale]['valid'].bool()
        final_params['level_multi']['valid'] = final_params['level_multi']['valid'].bool()
        return final_params
