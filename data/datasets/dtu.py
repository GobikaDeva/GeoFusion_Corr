"""DTU MVS dataset loader.

Per docs/baseline_recovery_plan.md, Stage 0 requires reproducing the baseline with the
EXACT SAME data split, view count, input scale, depth interval and augmentation as the
reference GeoMVS method -- so the `cfg["data"]` block should mirror that method's
published protocol (e.g. MVSNet/CasMVSNet-style DTU train/test/val split lists,
n_views, image size, depth interval and num_depth).

Confirmed against the attached DTU copy's actual layout:

    root/
      Cameras/{:08d}_cam.txt      # RAW 1600x1200 calibration -- NOT used (see _cam_path)
      Cameras/train/{:08d}_cam.txt  # MVSNet calibration at 160x128 (1/4 of 640x512) -- used
      Cameras/pair.txt            # per-view reference/source view selection
      Rectified/scan{N}_train/rect_{view+1:03d}_{light}_r5000.png
      Depths/scan{N}_train/depth_map_{view:04d}.pfm       # ground truth (train/val)
      Depths/scan{N}_train/depth_visual_{view:04d}.png    # visualization only, unused
      MonoPrior/scan{N}_train/{view:04d}_metric.npy        # metric geometric prior G (optional)

cfg["layout"] = "test" reads the MVSNet `dtu-test` layout instead -- full-resolution
1600x1200 images, the protocol every published DTU number (and GeoMVSNet's test.py)
is evaluated at:

    root/
      Cameras/{:08d}_cam.txt      # 1600x1200 calibration -- used
      Cameras/pair.txt
      Rectified/scan{N}/rect_{view+1:03d}_3_r5000.png
    prior_root/scan{N}/{view:04d}_metric.npy               # built with --layout test

Like GeoMVSNet's scale_mvs_input, the image is resized straight to img_wh (1600x1152
for the reference, i.e. the multiple-of-64 fit) with x/y intrinsics scaled separately.

Geometric prior (cfg["geometry_prior"]):
    "none"        ref_geom is all zeros (no prior).
    "mono_sparse" G = Depth-Anything-V2 monocular depth anchored to metric scale with
                  triangulated SIFT points and the known cameras (no GT used) -- built by
                  scripts/cache_mono_prior.py + scripts/anchor_mono_prior.py.
                  ref_geom = [normal_x, normal_y, normal_z, (G - depth_min) / sweep range],
                  and the sample also carries metric `prior_depth` / `prior_valid` for L_geo.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .camera_io import build_projection_matrix, read_cam_file
from .pfm_io import read_pfm


def depth_to_normals(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """(H, W) depth + (3, 3) intrinsic -> (3, H, W) unit camera-frame normals, oriented
    toward the camera (n_z < 0). Pixels with a zero-depth neighbour get a zero normal."""
    h, w = depth.shape
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], axis=0)  # (3, H, W)
    du = np.zeros_like(pts); dv = np.zeros_like(pts)
    du[:, :, 1:-1] = pts[:, :, 2:] - pts[:, :, :-2]
    dv[:, 1:-1, :] = pts[:, 2:, :] - pts[:, :-2, :]
    n = np.cross(du, dv, axis=0)
    n = n * np.where(n[2:3] > 0, -1.0, 1.0)
    norm = np.linalg.norm(n, axis=0, keepdims=True)
    # Central differences need the pixel and its 4 neighbours valid (border pixels never are).
    d = np.pad(depth > 0, 1, constant_values=False)
    ok = d[1:-1, 1:-1] & d[1:-1, :-2] & d[1:-1, 2:] & d[:-2, 1:-1] & d[2:, 1:-1]
    ok = ok[None] & (norm > 1e-8)
    return np.where(ok, n / np.maximum(norm, 1e-8), 0.0).astype(np.float32)


class DTUDataset(Dataset):
    """Standard DTU MVS dataset in MVSNet-family layout (see module docstring).

    Args:
        root: dataset root directory (contains Cameras/, Rectified/, Depths/).
        split: "train" | "val" | "test".
        cfg: dict with at least {n_views, img_wh, depth_interval, num_depth,
             scan_list_file (optional; defaults to data/datasets/splits/dtu_<split>.txt)}.

    Depth hypotheses: only the FIRST cascade stage's full-range sweep
    (depth_min .. depth_min + num_depth * depth_interval) is built here. Later stages
    are narrowed around the previous stage's predicted depth inside the model
    (models/geofusionnet.py, CasMVSNet-style), using `depth_interval` from
    model_inputs.
    """

    def __init__(self, root: str, split: str, cfg: dict):
        self.root = root
        self.split = split
        self.cfg = cfg
        self.n_views = cfg.get("n_views", 5)
        self.layout = cfg.get("layout", "train")
        if self.layout not in ("train", "test"):
            raise ValueError(f"Unknown DTU layout: {self.layout}")
        self.img_wh = tuple(cfg.get("img_wh", (640, 512)))
        self.num_depth = cfg.get("num_depth", 192)
        self.depth_interval_scale = cfg.get("depth_interval_scale", 1.0)
        self.lighting_conditions = cfg.get("lighting_conditions", list(range(7)))
        self.stage_downsample = cfg.get(
            "stage_downsample", {"coarse": 0.25, "mid": 0.5, "fine": 1.0}
        )
        # First cascade stage's full-range sweep; must match
        # models/geofusionnet.py's GeoFusionNetConfig.stages[0].
        self.first_stage = cfg.get("first_stage", "coarse")
        self.first_stage_num_depth = cfg.get("first_stage_num_depth", 48)

        # Resolution the Cameras/train intrinsics are expressed at (MVSNet stores them
        # at 1/4 of the 640x512 training images, i.e. 160x128).
        # The test layout's Cameras/*.txt are at the native 1600x1200.
        self.cam_calib_wh = tuple(cfg.get("cam_calib_wh", (160, 128) if self.layout == "train" else (1600, 1200)))

        self.geometry_prior = cfg.get("geometry_prior", "none")
        if self.geometry_prior not in ("none", "mono_sparse"):
            raise ValueError(f"Unknown geometry_prior: {self.geometry_prior}")
        self.prior_root = cfg.get("prior_root", os.path.join(root, "MonoPrior"))

        self.metas = self._build_metas()

    def _scan_list(self) -> list:
        list_file = self.cfg.get(
            "scan_list_file",
            os.path.join(os.path.dirname(__file__), "splits", f"dtu_{self.split}.txt"),
        )
        if not os.path.exists(list_file):
            raise FileNotFoundError(
                f"DTU {self.split} split list not found at {list_file}. "
                "Provide the standard MVSNet-style split list (see data/datasets/splits/README.md)."
            )
        with open(list_file) as f:
            return [line.strip() for line in f if line.strip()]

    def _build_metas(self) -> list:
        """Returns a list of (scan, ref_view, src_views) tuples using pair.txt view
        selection, matching the reference GeoMVS protocol exactly (Stage 0 requirement)."""
        metas = []
        pair_file = os.path.join(self.root, "Cameras", "pair.txt")
        if not os.path.exists(pair_file):
            return metas
        with open(pair_file) as f:
            num_viewpoints = int(f.readline())
            view_pairs = []
            for _ in range(num_viewpoints):
                ref_view = int(f.readline().rstrip())
                src_views_line = f.readline().rstrip().split()
                src_views = [int(x) for x in src_views_line[1::2]][: self.n_views - 1]
                if len(src_views) < self.n_views - 1:
                    continue  # not enough source views for this ref view; skip
                view_pairs.append((ref_view, src_views))
        for scan in self._scan_list():
            for ref_view, src_views in view_pairs:
                metas.append((scan, ref_view, src_views))
        return metas

    def __len__(self) -> int:
        return len(self.metas)

    def _scan_dir(self, scan: str) -> str:
        return f"{scan}_train" if self.layout == "train" else scan

    def _image_path(self, scan: str, view: int, light: int) -> str:
        return os.path.join(self.root, "Rectified", self._scan_dir(scan), f"rect_{view + 1:03d}_{light}_r5000.png")

    def _depth_path(self, scan: str, view: int) -> str:
        return os.path.join(self.root, "Depths", f"{scan}_train", f"depth_map_{view:04d}.pfm")

    def _cam_path(self, view: int) -> str:
        # Cameras/{:08d}_cam.txt holds the raw 1600x1200 DTU calibration, which does
        # not match the 640x512 Rectified images (those were downsampled 2x AND
        # cropped). Cameras/train/ holds the matching MVSNet calibration at 160x128.
        # The test layout's images are the full-res ones, so it uses the raw calibration.
        if self.layout == "test":
            return os.path.join(self.root, "Cameras", f"{view:08d}_cam.txt")
        return os.path.join(self.root, "Cameras", "train", f"{view:08d}_cam.txt")

    def _prior_path(self, scan: str, view: int) -> str:
        return os.path.join(self.prior_root, self._scan_dir(scan), f"{view:04d}_metric.npy")

    def _load_prior(self, scan: str, view: int, intrinsic: np.ndarray, depth_min: float, depth_max: float):
        """Returns (ref_geom (4, H, W), prior_depth (1, H, W) mm, prior_valid (1, H, W)).
        `intrinsic` must already be scaled to img_wh."""
        target_w, target_h = self.img_wh
        path = self._prior_path(scan, view)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Geometric prior not found: {path}. Build it with scripts/cache_mono_prior.py "
                "then scripts/anchor_mono_prior.py."
            )
        prior = np.load(path).astype(np.float32)
        valid = cv2.resize((prior > 0).astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
        prior = cv2.resize(prior, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        valid &= prior > 0
        prior = np.where(valid, prior, 0.0).astype(np.float32)

        normals = depth_to_normals(prior, intrinsic) * valid[None]
        prior_norm = np.where(valid, (prior - depth_min) / (depth_max - depth_min), 0.0)
        ref_geom = np.concatenate([normals, prior_norm[None]], axis=0).astype(np.float32)
        return (
            torch.from_numpy(ref_geom),
            torch.from_numpy(prior).unsqueeze(0),
            torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),
        )

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # (H, W, 3) uint8, native resolution

    def __getitem__(self, idx: int) -> dict:
        scan, ref_view, src_views = self.metas[idx]
        views = [ref_view] + list(src_views)

        light = random.choice(self.lighting_conditions) if self.split == "train" else 3

        raw_images = [self._load_image(self._image_path(scan, v, light)) for v in views]
        native_h, native_w = raw_images[0].shape[:2]
        target_w, target_h = self.img_wh
        # Intrinsics scale from the calibration's resolution, not the image file's.
        calib_w, calib_h = self.cam_calib_wh
        scale = (target_w / calib_w, target_h / calib_h)
        if self.layout == "train":
            assert abs(target_w / native_w - target_h / native_h) < 1e-3, (
                f"img_wh aspect ratio {self.img_wh} does not match native DTU image size "
                f"({native_w}x{native_h}); use an aspect-preserving img_wh."
            )
            assert abs(scale[0] - scale[1]) < 1e-3, (
                f"img_wh {self.img_wh} aspect ratio does not match cam_calib_wh {self.cam_calib_wh}"
            )
        else:
            # GeoMVSNet's scale_mvs_input: resize to the multiple-of-64 fit, anisotropically.
            assert (native_w, native_h) == (calib_w, calib_h), (
                f"test-layout image {native_w}x{native_h} does not match cam_calib_wh {self.cam_calib_wh}"
            )
            assert abs(scale[0] - scale[1]) < 0.05, f"img_wh {self.img_wh} distorts the image by >5%"

        images = []
        for img in raw_images:
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5  # [-1, 1]
            images.append(torch.from_numpy(normalized).permute(2, 0, 1))  # (3, H, W)

        cams = [read_cam_file(self._cam_path(v)) for v in views]
        # The model rescales per cascade stage internally (see
        # GeoFusionNet._scale_projection), so `scale` here only aligns the
        # calibration's resolution (cam_calib_wh) with the dataset's img_wh.
        projs = [
            torch.from_numpy(build_projection_matrix(c["intrinsic"], c["extrinsic"], scale=scale))
            for c in cams
        ]

        ref_cam = cams[0]
        depth_min = ref_cam["depth_min"]
        depth_interval = ref_cam["depth_interval"] * self.depth_interval_scale

        step = self.num_depth / self.first_stage_num_depth
        depth_hypotheses_per_stage = {
            self.first_stage: torch.from_numpy(
                depth_min + depth_interval * step * np.arange(self.first_stage_num_depth, dtype=np.float32)
            )
        }

        if self.geometry_prior == "none":
            ref_geom = torch.zeros(4, target_h, target_w)
            prior_depth = prior_valid = None
        else:
            ref_intrinsic = ref_cam["intrinsic"].copy()
            ref_intrinsic[0, :] *= scale[0]
            ref_intrinsic[1, :] *= scale[1]
            ref_geom, prior_depth, prior_valid = self._load_prior(
                scan, ref_view, ref_intrinsic, depth_min, depth_min + self.num_depth * depth_interval
            )

        sample = {
            "scan": scan,
            "ref_view": ref_view,
            "src_views": list(src_views),
            "model_inputs": {
                "ref_img": images[0],
                "src_imgs": images[1:],
                "ref_geom": ref_geom,
                "ref_proj": projs[0],
                "src_projs": projs[1:],
                "depth_hypotheses_per_stage": depth_hypotheses_per_stage,
                "depth_interval": torch.tensor(depth_interval, dtype=torch.float32),
            },
            "depth_min": depth_min,
            "depth_interval": depth_interval,
        }

        if prior_depth is not None:
            sample["prior_depth"] = prior_depth
            sample["prior_valid"] = prior_valid

        depth_path = self._depth_path(scan, ref_view)
        if self.split in ("train", "val") and os.path.exists(depth_path):
            gt_depth = read_pfm(depth_path)
            gt_depth_resized = cv2.resize(gt_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            sample["gt_depth"] = torch.from_numpy(gt_depth_resized).unsqueeze(0)
            sample["depth_valid_mask"] = (sample["gt_depth"] > 0).float()

        return sample
