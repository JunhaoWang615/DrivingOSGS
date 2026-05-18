import torch
from torch import nn
import torch.nn.functional as F


class HierarchicalGaussianSelector(nn.Module):
    """
    Local-structure complexity map construction + hierarchical Gaussian selection.

    Input image/depth shapes:
        - image: [B, 3, H, W] or [B, N, 3, H, W]
        - depth: [B, 1, H, W] or [B, N, 1, H, W]

    Gaussian dicts per level are expected to be spatial maps with the last two
    dimensions matching that level's [H_l, W_l].
    """

    def __init__(
        self,
        lambda_i=0.5,
        lambda_d=0.5,
        tau21=0.3,
        tau32=0.3,
        eps=1e-6,
    ):
        super().__init__()
        self.lambda_i = float(lambda_i)
        self.lambda_d = float(lambda_d)
        self.tau21 = float(tau21)
        self.tau32 = float(tau32)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _merge_bn(self, x):
        if x.dim() == 4:
            return x, False
        if x.dim() == 5:
            b, n = x.shape[:2]
            return x.reshape(b * n, *x.shape[2:]), True
        raise AssertionError(f"Expected 4D/5D tensor, got shape={tuple(x.shape)}")

    def _split_bn(self, x, original_shape, had_cam_dim):
        if not had_cam_dim:
            return x
        b, n = original_shape[:2]
        return x.reshape(b, n, *x.shape[1:])

    def _merge_bn_in_gaussian_dict(self, gaussian_dict, b, n):
        merged = {}
        for k, v in gaussian_dict.items():
            if not torch.is_tensor(v):
                merged[k] = v
                continue
            if v.dim() >= 5 and v.shape[0] == b and v.shape[1] == n:
                merged[k] = v.reshape(b * n, *v.shape[2:])
            else:
                merged[k] = v
        return merged

    def _to_gray(self, image_4d):
        assert image_4d.shape[1] == 3, "image channel must be 3"
        r = image_4d[:, 0:1, :, :]
        g = image_4d[:, 1:2, :, :]
        b = image_4d[:, 2:3, :, :]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _sobel_grad_mag(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def _minmax_normalize_per_sample(self, x, valid_mask=None):
        # x: [BN, 1, H, W]
        x_flat = x.flatten(1)

        if valid_mask is None:
            x_min = x_flat.min(dim=1, keepdim=True).values
            x_max = x_flat.max(dim=1, keepdim=True).values
            denom = (x_max - x_min).clamp_min(self.eps)
            x_norm = (x_flat - x_min) / denom
            return x_norm.view_as(x)

        valid = valid_mask.bool().flatten(1)
        neg_inf = torch.full_like(x_flat, -torch.inf)
        pos_inf = torch.full_like(x_flat, torch.inf)

        x_valid_max = torch.where(valid, x_flat, neg_inf).max(dim=1, keepdim=True).values
        x_valid_min = torch.where(valid, x_flat, pos_inf).min(dim=1, keepdim=True).values

        no_valid = (~valid).all(dim=1, keepdim=True)
        x_valid_max = torch.where(no_valid, torch.zeros_like(x_valid_max), x_valid_max)
        x_valid_min = torch.where(no_valid, torch.zeros_like(x_valid_min), x_valid_min)

        denom = (x_valid_max - x_valid_min).clamp_min(self.eps)
        x_norm = (x_flat - x_valid_min) / denom
        x_norm = x_norm.clamp(0.0, 1.0)
        x_norm = torch.where(valid, x_norm, torch.zeros_like(x_norm))
        return x_norm.view_as(x)

    def _assert_pyramid_shapes(self, c1, c2, c3):
        h1, w1 = c1.shape[-2:]
        h2, w2 = c2.shape[-2:]
        h3, w3 = c3.shape[-2:]
        assert h2 * 2 == h1 and w2 * 2 == w1, (
            f"Expected level_2 = level_1/2, got l1=({h1},{w1}), l2=({h2},{w2})"
        )
        assert h3 * 2 == h2 and w3 * 2 == w2, (
            f"Expected level_3 = level_2/2, got l2=({h2},{w2}), l3=({h3},{w3})"
        )

    def _assert_gaussian_spatial_shapes(self, g1, g2, g3):
        h1, w1 = self._infer_hw_from_gaussian_dict(g1)
        h2, w2 = self._infer_hw_from_gaussian_dict(g2)
        h3, w3 = self._infer_hw_from_gaussian_dict(g3)
        assert h2 * 2 == h1 and w2 * 2 == w1, (
            f"Expected gaussian l2=l1/2, got l1=({h1},{w1}), l2=({h2},{w2})"
        )
        assert h3 * 2 == h2 and w3 * 2 == w2, (
            f"Expected gaussian l3=l2/2, got l2=({h2},{w2}), l3=({h3},{w3})"
        )
        return (h1, w1), (h2, w2), (h3, w3)

    def _infer_hw_from_gaussian_dict(self, gaussian_dict):
        for _, v in gaussian_dict.items():
            if not torch.is_tensor(v) or v.dim() < 4:
                continue
            return v.shape[-2], v.shape[-1]
        raise AssertionError("Cannot infer (H, W) from gaussian dict")

    def _flatten_spatial(self, x):
        # [B, C, H, W] -> [B, H*W, C]
        assert x.dim() >= 4, f"Expected >=4D spatial tensor, got shape={tuple(x.shape)}"
        b = x.shape[0]
        h, w = x.shape[-2:]
        c_like = x.shape[1:-2]
        return x.reshape(b, -1, h * w).transpose(1, 2).reshape(b, h * w, *c_like)

    def _select_gaussians_by_mask(self, gaussian_dict, mask_bool):
        """
        Select gaussians by spatial bool mask.

        Args:
            gaussian_dict: dict[str, Tensor], tensors can be spatial maps with
                last two dims [H, W].
            mask_bool: [B, 1, H, W] bool

        Returns:
            dict with selected tensors concatenated across batch:
                - key -> [M, ...] (M is selected count over batch)
                - batch_index -> [M]
                - num_per_batch -> [B]
        """
        assert mask_bool.dtype == torch.bool, "mask must be bool"
        b, _, h, w = mask_bool.shape
        flat_mask = mask_bool.view(b, h * w)

        selected = {}
        batch_index = torch.arange(b, device=mask_bool.device, dtype=torch.long)
        batch_index = batch_index[:, None].expand(b, h * w)

        all_batch_idx = batch_index[flat_mask]
        selected["batch_index"] = all_batch_idx
        selected["num_per_batch"] = flat_mask.sum(dim=1)

        for k, v in gaussian_dict.items():
            if not torch.is_tensor(v):
                continue

            if v.dim() >= 4 and v.shape[-2:] == (h, w):
                v_flat = self._flatten_spatial(v)  # [B, HW, C...]
                kept = v_flat[flat_mask]
                selected[k] = kept
                continue

            if v.dim() >= 3 and v.shape[0] == b and v.shape[1] == h * w:
                kept = v[flat_mask]
                selected[k] = kept
                continue

            # Keep non-spatial fields unchanged to avoid accidental corruption.
            selected[k] = v

        return selected

    @staticmethod
    def _concat_selected_dicts(dict_list):
        out = {}
        common_keys = set(dict_list[0].keys())
        for d in dict_list[1:]:
            common_keys &= set(d.keys())

        for k in common_keys:
            vals = [d[k] for d in dict_list]
            if all(torch.is_tensor(v) for v in vals):
                ref = vals[0]
                can_cat = all(v.dim() >= 1 and v.shape[1:] == ref.shape[1:] for v in vals)
                if can_cat:
                    out[k] = torch.cat(vals, dim=0)
                else:
                    out[k] = vals
            else:
                out[k] = vals
        return out

    def forward(self, image, depth, gaussians_l1, gaussians_l2, gaussians_l3, valid_mask=None):
        image_4d, had_cam_dim = self._merge_bn(image)
        depth_4d, _ = self._merge_bn(depth)
        if had_cam_dim:
            b, n = image.shape[:2]
            gaussians_l1 = self._merge_bn_in_gaussian_dict(gaussians_l1, b, n)
            gaussians_l2 = self._merge_bn_in_gaussian_dict(gaussians_l2, b, n)
            gaussians_l3 = self._merge_bn_in_gaussian_dict(gaussians_l3, b, n)

        assert image_4d.shape[0] == depth_4d.shape[0], "image/depth batch mismatch"
        assert image_4d.shape[-2:] == depth_4d.shape[-2:], "image/depth resolution mismatch"

        if valid_mask is not None:
            valid_4d, _ = self._merge_bn(valid_mask)
            assert valid_4d.shape == depth_4d.shape, (
                f"valid_mask shape mismatch: valid={tuple(valid_4d.shape)} depth={tuple(depth_4d.shape)}"
            )
            valid_4d = valid_4d.bool()
        else:
            valid_4d = None

        # Build complexity C1 from grayscale and inverse-depth Sobel gradients.
        gray = self._to_gray(image_4d)
        gi = self._sobel_grad_mag(gray)

        inv_depth = 1.0 / (depth_4d.clamp_min(0.0) + self.eps)
        gd = self._sobel_grad_mag(inv_depth)

        gi_norm = self._minmax_normalize_per_sample(gi, valid_4d)
        gd_norm = self._minmax_normalize_per_sample(gd, valid_4d)

        c1 = self.lambda_i * gi_norm + self.lambda_d * gd_norm
        c2 = F.max_pool2d(c1, kernel_size=2, stride=2)
        c3 = F.max_pool2d(c2, kernel_size=2, stride=2)

        self._assert_pyramid_shapes(c1, c2, c3)
        (h1, w1), (h2, w2), (h3, w3) = self._assert_gaussian_spatial_shapes(
            gaussians_l1, gaussians_l2, gaussians_l3
        )

        assert c1.shape[-2:] == (h1, w1), f"C1 shape {tuple(c1.shape[-2:])} != l1 {(h1, w1)}"
        assert c2.shape[-2:] == (h2, w2), f"C2 shape {tuple(c2.shape[-2:])} != l2 {(h2, w2)}"
        assert c3.shape[-2:] == (h3, w3), f"C3 shape {tuple(c3.shape[-2:])} != l3 {(h3, w3)}"

        a21 = c2 < self.tau21
        a32 = c3 < self.tau32

        m3 = a32
        m3_up2 = F.interpolate(m3.float(), scale_factor=2.0, mode="nearest").bool()
        m2 = a21 & (~m3_up2)

        m2_up2 = F.interpolate(m2.float(), scale_factor=2.0, mode="nearest").bool()
        m3_up4 = F.interpolate(m3.float(), scale_factor=4.0, mode="nearest").bool()
        m1 = ~(m2_up2 | m3_up4)

        # Mutual exclusivity check in finest-space semantics.
        overlap_12 = m1 & m2_up2
        overlap_13 = m1 & m3_up4
        overlap_23 = m2_up2 & m3_up4
        assert not overlap_12.any(), "m1 and upsample(m2) overlap"
        assert not overlap_13.any(), "m1 and upsample(m3) overlap"
        assert not overlap_23.any(), "upsample(m2) and upsample(m3) overlap"

        # Enforce valid-mask if provided (per level by max-pool propagation).
        if valid_4d is not None:
            v1 = valid_4d
            v2 = F.max_pool2d(v1.float(), kernel_size=2, stride=2).bool()
            v3 = F.max_pool2d(v2.float(), kernel_size=2, stride=2).bool()
            m1 = m1 & v1
            m2 = m2 & v2
            m3 = m3 & v3

        sel_l1 = self._select_gaussians_by_mask(gaussians_l1, m1)
        sel_l2 = self._select_gaussians_by_mask(gaussians_l2, m2)
        sel_l3 = self._select_gaussians_by_mask(gaussians_l3, m3)
        sel_final = self._concat_selected_dicts([sel_l1, sel_l2, sel_l3])

        out = {
            "complexity": {
                "c1": self._split_bn(c1, image.shape, had_cam_dim),
                "c2": self._split_bn(c2, image.shape, had_cam_dim),
                "c3": self._split_bn(c3, image.shape, had_cam_dim),
            },
            "aggregation_masks": {
                "a21": self._split_bn(a21, image.shape, had_cam_dim),
                "a32": self._split_bn(a32, image.shape, had_cam_dim),
                "m1": self._split_bn(m1, image.shape, had_cam_dim),
                "m2": self._split_bn(m2, image.shape, had_cam_dim),
                "m3": self._split_bn(m3, image.shape, had_cam_dim),
            },
            "selected": {
                "l1": sel_l1,
                "l2": sel_l2,
                "l3": sel_l3,
                "final": sel_final,
            },
            "stats": {
                "num_l1": m1.flatten(1).sum(dim=1),
                "num_l2": m2.flatten(1).sum(dim=1),
                "num_l3": m3.flatten(1).sum(dim=1),
                "num_total": m1.flatten(1).sum(dim=1)
                + m2.flatten(1).sum(dim=1)
                + m3.flatten(1).sum(dim=1),
            },
        }
        return out
