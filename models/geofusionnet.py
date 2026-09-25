"""GeoFusionNet v2 backbone.

Design summary (docs/architecture_comparison.md):
    GeoFusionNet keeps separate RGB and geometry encoders, fuses them *before* the
    cost/plane-sweep volume is constructed (Geometry-Guided Feature, "GGF"), and
    regularizes the resulting cost volume in a coarse-to-fine cascade to produce a
    per-stage depth map.

This module intentionally mirrors a standard cascade-cost-volume MVSNet-family
architecture so that GeoCorr Lite (models/geocorr_lite.py) can hook into its matching
scores without needing a second geometry cost volume. Fill in the encoder/regularizer
bodies with the exact GeoFusionNet v2 spec when integrating real code; the interfaces
(feature extraction -> plane-sweep warp -> cost volume -> per-stage matching scores)
are what the rest of this repository depends on and should be kept stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class CascadeStageConfig:
    name: str
    num_depth_hypotheses: int
    resolution_scale: float  # relative to full input resolution, e.g. 0.25, 0.5, 1.0
    # Hypothesis spacing as a multiple of the camera's depth_interval. The first
    # stage sweeps the full range; later stages sweep num_depth_hypotheses *
    # depth_interval_ratio around the previous stage's upsampled depth (CasMVSNet).
    depth_interval_ratio: float = 1.0
    channels: int = 32


@dataclass
class GeoFusionNetConfig:
    in_channels: int = 3
    geometry_channels: int = 4  # e.g. normal (3) + prior depth (1)
    base_channels: int = 32
    stages: List[CascadeStageConfig] = field(default_factory=lambda: [
        CascadeStageConfig("coarse", num_depth_hypotheses=48, resolution_scale=0.25, depth_interval_ratio=4.0),
        CascadeStageConfig("mid", num_depth_hypotheses=32, resolution_scale=0.5, depth_interval_ratio=2.0),
        CascadeStageConfig("fine", num_depth_hypotheses=8, resolution_scale=1.0, depth_interval_ratio=1.0),
    ])
    use_ggf_residual: bool = False       # Stage 1: zero-initialized GGF residual
    ggf_stage_index: int = 0             # apply GGF at one low-resolution feature level
    # Exclude a source view from the cost-volume variance at (depth, pixel) samples
    # that project outside its image, instead of counting the zero-padded feature
    # as a real observation. Samples seen by <2 views get the pixel's mean cost over
    # its valid depths (flat, uninformative).
    mask_out_of_view: bool = False
    # 3D regularizer upsampling: "deconv" (ConvTranspose3d, original), "deconv_smooth"
    # (plus a fixed [1,2,1] blur that removes stride-2 checkerboarding; no new
    # parameters), or "trilinear" (trilinear upsample + 3x3x3 conv; new weights).
    reg_upsample: str = "deconv"


class RGBEncoder(nn.Module):
    """Shared multi-scale RGB feature pyramid (one branch per view, weight-shared)."""

    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        f0 = self.stem(x)       # full res
        f1 = self.down1(f0)     # 1/2 res
        f2 = self.down2(f1)     # 1/4 res
        return {"fine": f0, "mid": f1, "coarse": f2}


class GeometryEncoder(nn.Module):
    """Encodes normal / prior-depth geometry maps at matching resolutions.

    Kept separate from RGBEncoder per the "shared design principle": geometry and RGB
    are not concatenated at the first layer.
    """

    def __init__(self, geometry_channels: int, base_channels: int):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(geometry_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
        )

    def forward(self, geom: torch.Tensor):
        g0 = self.stem(geom)
        g1 = self.down1(g0)
        g2 = self.down2(g1)
        return {"fine": g0, "mid": g1, "coarse": g2}


class GGFResidualFusion(nn.Module):
    """Geometry-Guided Feature fusion, applied as a zero-initialized residual.

    Stage 1 of the baseline recovery plan: `use_ggf_residual=True` with the final
    projection zero-initialized so that, at the start of training, this block is
    exactly an identity on the RGB feature (baseline forward path preserved).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # Zero-init the final layer -> residual starts as a no-op.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, rgb_feat: torch.Tensor, geom_feat: torch.Tensor) -> torch.Tensor:
        residual = self.proj(torch.cat([rgb_feat, geom_feat], dim=1))
        return rgb_feat + residual


def _blur121_3d(x: torch.Tensor) -> torch.Tensor:
    """Separable [1, 2, 1] / 4 blur along D, H and W (replicate-padded)."""
    k = x.new_tensor([0.25, 0.5, 0.25])
    C = x.shape[1]
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = 3
        pad = [0] * 6
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = 1
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape).expand(C, 1, *shape[2:]).contiguous(), groups=C)
    return x


class CostRegularizer3D(nn.Module):
    """Lightweight 3D U-Net style regularizer over the (D, H, W) cost volume."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8, upsample: str = "deconv"):
        super().__init__()
        if upsample not in ("deconv", "deconv_smooth", "trilinear"):
            raise ValueError(f"Unknown reg_upsample: {upsample}")
        self.upsample = upsample
        c = base_channels
        self.conv0 = nn.Sequential(nn.Conv3d(in_channels, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(c, c * 2, 3, stride=2, padding=1), nn.BatchNorm3d(c * 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(c * 2, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        if upsample == "trilinear":
            self.up = nn.Conv3d(c, c, 3, padding=1)
        else:
            self.up = nn.ConvTranspose3d(c, c, kernel_size=2, stride=2)
        self.out = nn.Conv3d(c, 1, 3, padding=1)

    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        x0 = self.conv0(cost_volume)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        if self.upsample == "trilinear":
            x2 = self.up(F.interpolate(x2, size=x0.shape[-3:], mode="trilinear", align_corners=False))
        else:
            x2 = self.up(x2)
            if self.upsample == "deconv_smooth":
                x2 = _blur121_3d(x2)
        if x2.shape[-3:] != x0.shape[-3:]:
            x2 = F.interpolate(x2, size=x0.shape[-3:], mode="trilinear", align_corners=False)
        return self.out(x0 + x2).squeeze(1)  # (B, D, H, W) matching score volume


def differentiable_homography_warp(
    src_feat: torch.Tensor,
    src_proj: torch.Tensor,
    ref_proj: torch.Tensor,
    depth_hypotheses: torch.Tensor,
    return_valid: bool = False,
):
    """Plane-sweep warp of a source feature map into the reference view.

    This is the canonical MVSNet-family plane-sweep homography warp. It is the SAME
    projection grid that GeoCorr Lite's scalar cues (models/geometry_cues.py) must
    reuse -- do not build a second grid for the geometry cues.

    `src_proj` and `ref_proj` are full 4x4 projection matrices (intrinsic @ extrinsic,
    with the last row [0, 0, 0, 1]) -- see data/datasets/camera_io.build_projection_matrix,
    which must be called with the SAME resize scale as `src_feat`'s resolution
    (i.e. built per cascade stage, not once at full resolution).

    Args:
        src_feat: (B, C, H, W) source feature map.
        src_proj: (B, 4, 4) source camera projection matrix, matching src_feat's resolution.
        ref_proj: (B, 4, 4) reference camera projection matrix, matching src_feat's resolution.
        depth_hypotheses: (B, D) plane-sweep depths shared by every reference pixel, or
            (B, D, H, W) per-pixel depths (cascade stages narrowed around a previous
            stage's estimate).

    Returns:
        (B, C, D, H, W) warped source features, one slice per depth hypothesis; with
        `return_valid`, also a (B, 1, D, H, W) float mask of samples that land inside
        the source image in front of the camera.
    """
    B, C, H, W = src_feat.shape
    D = depth_hypotheses.shape[1]
    per_pixel = depth_hypotheses.dim() == 4

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))  # (B, 4, 4)
        rot = proj[:, :3, :3]      # (B, 3, 3)
        trans = proj[:, :3, 3:4]  # (B, 3, 1)

        y, x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=src_feat.device),
            torch.arange(0, W, dtype=torch.float32, device=src_feat.device),
            indexing="ij",
        )
        y, x = y.reshape(-1), x.reshape(-1)  # (H*W,)
        xyz = torch.stack((x, y, torch.ones_like(x)), dim=0)  # (3, H*W)
        xyz = xyz.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)

        rot_xyz = torch.matmul(rot, xyz)  # (B, 3, H*W)
        depth_view = depth_hypotheses.reshape(B, 1, D, H * W) if per_pixel else depth_hypotheses.view(B, 1, D, 1)
        rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_view  # (B, 3, D, H*W)
        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)  # (B, 3, D, H*W)

        z = proj_xyz[:, 2:3, :, :].clamp(min=1e-6)
        proj_xy = proj_xyz[:, :2, :, :] / z  # (B, 2, D, H*W)

        proj_x_norm = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1
        proj_y_norm = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        grid = torch.stack((proj_x_norm, proj_y_norm), dim=3)  # (B, D, H*W, 2)
        grid = grid.view(B, D * H, W, 2)

    warped = F.grid_sample(
        src_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # (B, C, D*H, W)
    warped = warped.view(B, C, D, H, W)
    if not return_valid:
        return warped
    with torch.no_grad():
        valid = (
            (proj_x_norm.abs() <= 1) & (proj_y_norm.abs() <= 1) & (proj_xyz[:, 2, :, :] > 1e-3)
        ).float().view(B, 1, D, H, W)
    return warped, valid


def regress_depth(score_volume: torch.Tensor, depth_hypotheses: torch.Tensor) -> torch.Tensor:
    """Soft-argmin depth regression: converts a (B, D, H, W) matching-score volume
    (HIGHER_IS_BETTER convention, matching models.geocorr_lite.DEFAULT_CONVENTION)
    into a (B, 1, H, W) continuous depth map by taking the expectation of
    `depth_hypotheses` under softmax(score_volume) along the depth dimension.
    `depth_hypotheses` is (B, D) or per-pixel (B, D, H, W).
    """
    B, D, H, W = score_volume.shape
    prob = F.softmax(score_volume, dim=1)  # (B, D, H, W)
    depth_hyp = depth_hypotheses if depth_hypotheses.dim() == 4 else depth_hypotheses.view(B, D, 1, 1)
    depth = (prob * depth_hyp).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    return depth


class GeoFusionNet(nn.Module):
    """Cascade cost-volume MVS network with optional zero-initialized GGF residual.

    Forward returns, per cascade stage, the raw matching-score volume BEFORE softmax
    regression, so that GeoCorr Lite can add its residual to the score prior to
    depth regression (see models/geocorr_lite.py and models/build_model.py).
    """

    def __init__(self, cfg: GeoFusionNetConfig):
        super().__init__()
        self.cfg = cfg
        self.rgb_encoder = RGBEncoder(cfg.in_channels, cfg.base_channels)
        self.geometry_encoder = GeometryEncoder(cfg.geometry_channels, cfg.base_channels)

        self.ggf_fusion = None
        if cfg.use_ggf_residual:
            stage_channels = {
                "fine": cfg.base_channels,
                "mid": cfg.base_channels * 2,
                "coarse": cfg.base_channels * 4,
            }
            stage_name = cfg.stages[cfg.ggf_stage_index].name
            self.ggf_fusion = GGFResidualFusion(stage_channels[stage_name])
            self.ggf_stage_name = stage_name

        self.regularizers = nn.ModuleDict({
            stage.name: CostRegularizer3D(upsample=cfg.reg_upsample) for stage in cfg.stages
        })

    COST_VOLUME_DEPTH_CHUNK = 8

    def build_cost_volume(
        self,
        ref_feat: torch.Tensor,
        src_feats: List[torch.Tensor],
        ref_proj: torch.Tensor,
        src_projs: List[torch.Tensor],
        depth_hypotheses: torch.Tensor,
    ) -> torch.Tensor:
        """Variance-based cost volume from ref/src warped features, averaged over channels.

        The unbiased variance across views is accumulated from running sums rather
        than by stacking every warped volume, so only a couple of (B, C, D, H, W)
        buffers are alive at once. Without autograd (inference), depth hypotheses are
        processed in chunks of COST_VOLUME_DEPTH_CHUNK -- each depth is independent, so
        the result is identical -- which is what lets 1600x1152 DTU testing fit in 24GB.
        """
        chunk = self.COST_VOLUME_DEPTH_CHUNK
        if not torch.is_grad_enabled() and depth_hypotheses.shape[1] > chunk:
            parts = [
                self._cost_volume_chunk(ref_feat, src_feats, ref_proj, src_projs, depth_hypotheses[:, d:d + chunk])
                for d in range(0, depth_hypotheses.shape[1], chunk)
            ]
            volume = torch.cat([p[0] for p in parts], dim=2)
            n = torch.cat([p[1] for p in parts], dim=2) if self.cfg.mask_out_of_view else None
        else:
            volume, n = self._cost_volume_chunk(ref_feat, src_feats, ref_proj, src_projs, depth_hypotheses)
        if n is not None:
            ok = n >= 2
            fill = (volume * ok).sum(dim=2, keepdim=True) / ok.sum(dim=2, keepdim=True).clamp(min=1)
            volume = torch.where(ok, volume, fill)
        return volume

    def _cost_volume_chunk(self, ref_feat, src_feats, ref_proj, src_projs, depth_hypotheses):
        """Returns ((B, 1, D, H, W) channel-mean variance, per-sample view count or None)."""
        ref_expanded = ref_feat.unsqueeze(2).expand(-1, -1, depth_hypotheses.shape[1], -1, -1)
        vol_sum = ref_expanded
        vol_sq_sum = ref_expanded ** 2
        if not self.cfg.mask_out_of_view:
            for sf, sp in zip(src_feats, src_projs):
                warped = differentiable_homography_warp(sf, sp, ref_proj, depth_hypotheses)
                vol_sum = vol_sum + warped
                vol_sq_sum = vol_sq_sum + warped ** 2
                del warped
            n = len(src_feats) + 1
            volume = (vol_sq_sum - vol_sum ** 2 / n) / (n - 1)  # (B, C, D, H, W), == stack(...).var(dim=0)
            return volume.mean(dim=1, keepdim=True), None  # (B, 1, D, H, W) group-wise reduced
        n = torch.ones_like(ref_expanded[:, :1])
        for sf, sp in zip(src_feats, src_projs):
            warped, valid = differentiable_homography_warp(sf, sp, ref_proj, depth_hypotheses, return_valid=True)
            vol_sum = vol_sum + warped * valid
            vol_sq_sum = vol_sq_sum + warped ** 2 * valid
            n = n + valid
            del warped
        volume = (vol_sq_sum - vol_sum ** 2 / n) / (n - 1).clamp(min=1)  # unbiased variance over valid views
        return volume.mean(dim=1, keepdim=True), n

    @staticmethod
    def _scale_projection(proj: torch.Tensor, scale: float) -> torch.Tensor:
        """Rescales a full-resolution 4x4 projection matrix (see
        data/datasets/camera_io.build_projection_matrix) to match a feature map
        downsampled by `scale` relative to the full-resolution image. Only the x/y
        (row 0, 1) components of the intrinsic scale with image size; the
        depth/homogeneous row does not.
        """
        if scale == 1.0:
            return proj
        scaled = proj.clone()
        scaled[:, :2, :] = scaled[:, :2, :] * scale
        return scaled

    @staticmethod
    def _narrowed_hypotheses(
        prev_depth: torch.Tensor, size: tuple, num_depth: int, interval: torch.Tensor
    ) -> torch.Tensor:
        """Per-pixel hypotheses centred on the previous stage's (detached, upsampled)
        depth: num_depth samples spaced `interval` apart (CasMVSNet-style).

        prev_depth: (B, 1, h, w); interval: (B,). Returns (B, num_depth, H, W).
        """
        center = F.interpolate(prev_depth.detach(), size=size, mode="bilinear", align_corners=False)
        step = interval.view(-1, 1, 1, 1).to(center.dtype)
        start = (center - (num_depth / 2) * step).clamp(min=1e-3)
        offsets = torch.arange(num_depth, device=center.device, dtype=center.dtype).view(1, -1, 1, 1)
        return start + offsets * step

    def forward(
        self,
        ref_img: torch.Tensor,
        src_imgs: List[torch.Tensor],
        ref_geom: torch.Tensor,
        ref_proj: torch.Tensor,
        src_projs: List[torch.Tensor],
        depth_hypotheses_per_stage: dict,
        depth_interval: Optional[torch.Tensor] = None,
        stages: Optional[List[str]] = None,
    ) -> dict:
        """Returns {"scores": {stage: raw score volume (B, D, H, W)},
                    "depth_hypotheses": {stage: per-pixel hypotheses (B, D, H, W)}}.

        `ref_proj`/`src_projs` must be built at FULL input resolution (scale=1.0 in
        data/datasets/camera_io.build_projection_matrix); this method rescales them
        per cascade stage to match each stage's feature-map resolution.

        Cascade narrowing: only the first stage reads `depth_hypotheses_per_stage`
        (its full-range sweep). Each later stage sweeps num_depth_hypotheses samples,
        depth_interval * depth_interval_ratio apart, around the previous stage's
        regressed depth. `depth_interval` (B,) is the camera's base interval.

        `stages`: if given, stages are computed in order up to and including the
        deepest one listed (later stages depend on earlier ones). None = all stages.
        """
        ref_feats = self.rgb_encoder(ref_img)
        src_feats_list = [self.rgb_encoder(s) for s in src_imgs]
        geom_feats = self.geometry_encoder(ref_geom)

        if self.ggf_fusion is not None:
            stage_name = self.ggf_stage_name
            ref_feats[stage_name] = self.ggf_fusion(ref_feats[stage_name], geom_feats[stage_name])

        active_stages = self.cfg.stages
        if stages is not None:
            last = max(i for i, s in enumerate(self.cfg.stages) if s.name in stages)
            active_stages = self.cfg.stages[: last + 1]

        scores, hypotheses = {}, {}
        prev_depth = None
        for i, stage in enumerate(active_stages):
            ref_f = ref_feats[stage.name]
            src_fs = [sf[stage.name] for sf in src_feats_list]
            H, W = ref_f.shape[-2:]
            if i == 0:
                depth_hyp = depth_hypotheses_per_stage[stage.name]
                depth_hyp = depth_hyp.view(*depth_hyp.shape, 1, 1).expand(-1, -1, H, W)
            else:
                if depth_interval is None:
                    raise ValueError("depth_interval is required for cascade narrowing of later stages")
                depth_hyp = self._narrowed_hypotheses(
                    prev_depth, (H, W), stage.num_depth_hypotheses, depth_interval * stage.depth_interval_ratio
                )
            stage_ref_proj = self._scale_projection(ref_proj, stage.resolution_scale)
            stage_src_projs = [self._scale_projection(sp, stage.resolution_scale) for sp in src_projs]
            if self.training and torch.is_grad_enabled():
                # Recompute the cost volume in backward instead of storing it (the
                # full-cascade volumes don't fit in 24GB at batch 2 otherwise).
                cost_volume = checkpoint(
                    self.build_cost_volume, ref_f, src_fs, stage_ref_proj, stage_src_projs, depth_hyp,
                    use_reentrant=False,
                )
            else:
                cost_volume = self.build_cost_volume(ref_f, src_fs, stage_ref_proj, stage_src_projs, depth_hyp)
            scores[stage.name] = self.regularizers[stage.name](cost_volume)
            hypotheses[stage.name] = depth_hyp
            prev_depth = regress_depth(scores[stage.name], depth_hyp)
        return {"scores": scores, "depth_hypotheses": hypotheses}
