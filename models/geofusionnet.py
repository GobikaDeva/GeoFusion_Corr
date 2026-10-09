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
    # Capacity options (defaults = the original architecture, which A0f uses).
    # encoder: "basic" (one conv per level) or "fpn" (wider bottom-up path with
    # fpn_widths channels, 2-3 convs per level, plus a top-down FPN path; outputs keep
    # the basic encoder's base_channels * (1, 2, 4) so the cost volume shapes match).
    encoder: str = "basic"
    fpn_widths: List[int] = field(default_factory=lambda: [32, 96, 192])
    fpn_inner: int = 64  # top-down path width (it runs at full resolution for 5 views)
    # Cost-volume channels: 1 = variance averaged over all feature channels (original);
    # G > 1 = group-wise variance, averaged within G equal channel groups (G channels
    # into the regularizer).
    cost_groups: int = 1
    # 3D regularizer: reg_levels 1 = the original CostRegularizer3D (base 8);
    # 2 = CostRegularizerUNet3D with two down/up levels (trilinear upsampling only).
    reg_base_channels: int = 8
    reg_levels: int = 1
    # Frozen monocular features (B2_mono_feat): Depth-Anything-V2-Small's fused DPT
    # feature of the REFERENCE image, projected by a zero-initialized 1x1 conv and added
    # to the reference encoder features at mono_feat_stages. Source features unchanged.
    mono_features: bool = False
    mono_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    mono_feat_stages: List[str] = field(default_factory=lambda: ["coarse", "mid"])
    mono_input_h: int = 518   # Depth-Anything input height (multiple of its 14px patch)
    mono_fp16: bool = True    # run the frozen network under fp16 autocast on CUDA


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


def _conv_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FPNEncoder(nn.Module):
    """Wider 3-level RGB pyramid with a top-down FPN path (CasMVSNet FeatureNet style).

    Bottom-up: `widths` channels at 1, 1/2, 1/4 resolution (2, 3, 3 convs). Top-down:
    1x1 laterals into `inner` channels, nearest-upsampled and summed, then a
    3x3 output conv per level to out_channels * (1, 2, 4) for fine / mid / coarse.
    """

    def __init__(self, in_channels: int, out_channels: int, widths: List[int], inner: int):
        super().__init__()
        w0, w1, w2 = widths
        self.level0 = nn.Sequential(_conv_bn_relu(in_channels, w0), _conv_bn_relu(w0, w0))
        self.level1 = nn.Sequential(_conv_bn_relu(w0, w1, 2), _conv_bn_relu(w1, w1), _conv_bn_relu(w1, w1))
        self.level2 = nn.Sequential(_conv_bn_relu(w1, w2, 2), _conv_bn_relu(w2, w2), _conv_bn_relu(w2, w2))
        self.lat2 = nn.Conv2d(w2, inner, 1)
        self.lat1 = nn.Conv2d(w1, inner, 1)
        self.lat0 = nn.Conv2d(w0, inner, 1)
        self.out2 = nn.Conv2d(inner, out_channels * 4, 3, padding=1)
        self.out1 = nn.Conv2d(inner, out_channels * 2, 3, padding=1)
        self.out0 = nn.Conv2d(inner, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        c0 = self.level0(x)
        c1 = self.level1(c0)
        c2 = self.level2(c1)
        p2 = self.lat2(c2)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        if self.training and torch.is_grad_enabled():
            # full-res top-down step recomputed in backward (no BatchNorm in it, so exact)
            fine = checkpoint(self._fine, c0, p1, use_reentrant=False)
        else:
            fine = self._fine(c0, p1)
        return {"fine": fine, "mid": self.out1(p1), "coarse": self.out2(p2)}

    def _fine(self, c0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        return self.out0(self.lat0(c0) + F.interpolate(p1, size=c0.shape[-2:], mode="nearest"))


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


class MonoFeatureInjection(nn.Module):
    """Frozen Depth-Anything-V2 features of the reference image as a zero-initialized
    residual on the reference encoder features.

    The frozen network runs under no_grad (fp16 autocast on CUDA) and is held outside
    the module tree, so it is not in parameters() (optimizer) or state_dict()
    (checkpoints stay B2-compatible); _apply moves/casts it with the module. Its last
    fused DPT neck map (64 ch, 8/14 of its input resolution) is resized to each stage
    and projected by a 1x1 conv whose weight and bias start at zero, so at
    initialization the reference features are unchanged.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id: str, stage_channels: dict, input_h: int, fp16: bool):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        da = AutoModelForDepthEstimation.from_pretrained(model_id).eval()
        da.requires_grad_(False)
        self._frozen = [da]  # a list keeps it out of the module tree
        self.patch = da.config.patch_size
        self.input_h = input_h
        self.fp16 = fp16
        c_in = da.config.fusion_hidden_size
        self.proj = nn.ModuleDict({name: nn.Conv2d(c_in, c, 1) for name, c in stage_channels.items()})
        for conv in self.proj.values():
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1), persistent=False)

    def _apply(self, fn, recurse=True):
        self._frozen[0]._apply(fn)
        return super()._apply(fn, recurse)

    @torch.no_grad()
    def mono_feature(self, ref_img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) image in [-1, 1] (data/datasets/dtu.py) -> (B, C, h', w') fp32."""
        da = self._frozen[0]
        H, W = ref_img.shape[-2:]
        h = self.input_h
        w = max(1, round(W * h / H / self.patch)) * self.patch
        x = F.interpolate(ref_img * 0.5 + 0.5, size=(h, w), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=self.fp16 and x.is_cuda):
            hidden = da.backbone.forward_with_filtered_kwargs(x).feature_maps
            fused = da.neck(hidden, h // self.patch, w // self.patch)[-1]
        return fused.float()

    def forward(self, ref_img: torch.Tensor, sizes: dict) -> dict:
        """{stage: (h, w)} -> {stage: (B, C_stage, h, w) residual}."""
        feat = self.mono_feature(ref_img)
        return {
            name: self.proj[name](F.interpolate(feat, size=tuple(size), mode="bilinear", align_corners=False, antialias=True))
            for name, size in sizes.items()
        }


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


def _conv3d_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


class CostRegularizerUNet3D(nn.Module):
    """Larger 3D U-Net regularizer: `levels` stride-2 down/up levels, channels
    base * 2**level, trilinear upsampling + 3x3x3 conv, additive skips."""

    def __init__(self, in_channels: int, base_channels: int = 16, levels: int = 2):
        super().__init__()
        c = base_channels
        self.levels = levels
        self.conv0 = _conv3d_bn_relu(in_channels, c)
        self.down = nn.ModuleList([
            nn.Sequential(_conv3d_bn_relu(c * 2 ** i, c * 2 ** (i + 1), 2), _conv3d_bn_relu(c * 2 ** (i + 1), c * 2 ** (i + 1)))
            for i in range(levels)
        ])
        self.up = nn.ModuleList([_conv3d_bn_relu(c * 2 ** (i + 1), c * 2 ** i) for i in range(levels)])
        self.out = nn.Conv3d(c, 1, 3, padding=1)

    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        skips = [self.conv0(cost_volume)]
        for down in self.down:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for i in reversed(range(self.levels)):
            skip = skips.pop()
            x = skip + self.up[i](F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False))
        return self.out(x).squeeze(1)  # (B, D, H, W) matching score volume


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
        if cfg.encoder == "basic":
            self.rgb_encoder = RGBEncoder(cfg.in_channels, cfg.base_channels)
        elif cfg.encoder == "fpn":
            self.rgb_encoder = FPNEncoder(cfg.in_channels, cfg.base_channels, cfg.fpn_widths, cfg.fpn_inner)
        else:
            raise ValueError(f"Unknown encoder: {cfg.encoder}")
        if any((cfg.base_channels * k) % cfg.cost_groups for k in (1, 2, 4)):
            raise ValueError(f"cost_groups {cfg.cost_groups} must divide every stage's feature channels")
        self.geometry_encoder = GeometryEncoder(cfg.geometry_channels, cfg.base_channels)
        stage_channels = {
            "fine": cfg.base_channels,
            "mid": cfg.base_channels * 2,
            "coarse": cfg.base_channels * 4,
        }

        self.mono_injection = None
        if cfg.mono_features:
            self.mono_injection = MonoFeatureInjection(
                cfg.mono_model_id, {s: stage_channels[s] for s in cfg.mono_feat_stages}, cfg.mono_input_h, cfg.mono_fp16
            )

        self.ggf_fusion = None
        if cfg.use_ggf_residual:
            stage_name = cfg.stages[cfg.ggf_stage_index].name
            self.ggf_fusion = GGFResidualFusion(stage_channels[stage_name])
            self.ggf_stage_name = stage_name

        if cfg.reg_levels == 1:
            make_reg = lambda: CostRegularizer3D(cfg.cost_groups, cfg.reg_base_channels, upsample=cfg.reg_upsample)  # noqa: E731
        else:
            if cfg.reg_upsample != "trilinear":
                raise ValueError("reg_levels > 1 supports reg_upsample: trilinear only")
            make_reg = lambda: CostRegularizerUNet3D(cfg.cost_groups, cfg.reg_base_channels, cfg.reg_levels)  # noqa: E731
        self.regularizers = nn.ModuleDict({stage.name: make_reg() for stage in cfg.stages})

    COST_VOLUME_DEPTH_CHUNK = 8

    def build_cost_volume(
        self,
        ref_feat: torch.Tensor,
        src_feats: List[torch.Tensor],
        ref_proj: torch.Tensor,
        src_projs: List[torch.Tensor],
        depth_hypotheses: torch.Tensor,
    ) -> torch.Tensor:
        """Variance-based cost volume from ref/src warped features, averaged over channels
        (or within cfg.cost_groups channel groups): (B, cost_groups, D, H, W).

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
        """Returns ((B, G, D, H, W) group-mean variance, per-sample view count or None)."""
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
            return self._group_mean(volume), None  # (B, G, D, H, W) group-wise reduced
        n = torch.ones_like(ref_expanded[:, :1])
        for sf, sp in zip(src_feats, src_projs):
            warped, valid = differentiable_homography_warp(sf, sp, ref_proj, depth_hypotheses, return_valid=True)
            vol_sum = vol_sum + warped * valid
            vol_sq_sum = vol_sq_sum + warped ** 2 * valid
            n = n + valid
            del warped
        volume = (vol_sq_sum - vol_sum ** 2 / n) / (n - 1).clamp(min=1)  # unbiased variance over valid views
        return self._group_mean(volume), n

    def _group_mean(self, volume: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) -> (B, G, D, H, W), mean within G equal channel groups."""
        G = self.cfg.cost_groups
        if G == 1:
            return volume.mean(dim=1, keepdim=True)
        B, C = volume.shape[:2]
        return volume.view(B, G, C // G, *volume.shape[2:]).mean(dim=2)

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

    def extract_features(self, ref_img: torch.Tensor, src_imgs: List[torch.Tensor], ref_geom: torch.Tensor):
        """Per-stage features that enter the cost volume: ({stage: ref feature},
        [{stage: src feature}] per source view). The reference features include the
        mono-feature and GGF residuals when enabled; source features are RGB-only.
        Anything that rebuilds a cost volume outside forward() (e.g.
        scripts/costvol_profile_probe.py) must use this, not rgb_encoder directly."""
        ref_feats = self.rgb_encoder(ref_img)
        src_feats_list = [self.rgb_encoder(s) for s in src_imgs]
        if self.mono_injection is not None:
            sizes = {name: ref_feats[name].shape[-2:] for name in self.cfg.mono_feat_stages}
            for name, residual in self.mono_injection(ref_img, sizes).items():
                ref_feats[name] = ref_feats[name] + residual
        if self.ggf_fusion is not None:
            stage_name = self.ggf_stage_name
            geom_feats = self.geometry_encoder(ref_geom)
            ref_feats[stage_name] = self.ggf_fusion(ref_feats[stage_name], geom_feats[stage_name])
        return ref_feats, src_feats_list

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
        ref_feats, src_feats_list = self.extract_features(ref_img, src_imgs, ref_geom)

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
