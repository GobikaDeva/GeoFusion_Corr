#!/usr/bin/env python
"""CLI entry point: train one stage/ablation config for one or more seeds.

Usage:
    python -m training.train --config configs/stage0_baseline.yaml --seed 0
    python -m training.train --config configs/ablations/A3_geocorr_fixed.yaml --seed 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models import (
    build_model,
    cascade_photometric_confidence,
    compact_self_supervised_loss,
    consistency_depth_loss,
    regress_depth,
    supervised_l1_loss,
)
from training.schedule import AlphaSchedule, GateCeilingSchedule, PriorDropoutSchedule
from evaluation.stage_depth import sample_stage_stats, summarize
from training.provenance import log_git_commit
from training.seed_utils import set_seed
from training.trainer import Trainer, TrainerConfig


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_dataloader(cfg: dict, split: str):
    """See data/datasets/dtu.py -- wire this up once the real DTU (or generalization)
    dataset root is available. Kept out of this scaffold's critical path so the
    model/schedule/gate logic can be tested independently (see tests/)."""
    from data.datasets.dtu import DTUDataset
    ds = DTUDataset(root=cfg["data"]["root"], split=split, cfg=cfg["data"])
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg["training"].get("batch_size", 2), shuffle=(split == "train"),
        num_workers=cfg["data"].get("num_workers", 4),
    )


def move_batch_to_device(batch, device):
    """Recursively moves a (possibly nested) dict/list of tensors onto `device`,
    leaving non-tensor entries (scan names, view-id lists, etc.) untouched."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    return batch


DEFAULT_STAGE_WEIGHTS = {"coarse": 0.5, "mid": 1.0, "fine": 2.0}  # CasMVSNet's per-stage weights


def _scale_proj(proj: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return proj
    scaled = proj.clone()
    scaled[:, :2, :] = scaled[:, :2, :] * scale
    return scaled


def make_loss_fn(cfg: dict):
    """Builds the loss function specified in the stage config's `loss` section, per
    docs/baseline_recovery_plan.md ('keep the baseline objective unchanged during the
    first GeoCorr Lite experiment').

    With cascade narrowing every stage feeds the next, so the same objective is
    applied to each stage's regressed depth at that stage's resolution and summed
    with `loss.stage_weights` (default CasMVSNet 0.5/1/2 for coarse/mid/fine).
    """
    loss_cfg = cfg.get("loss", {"type": "compact_self_supervised"})
    loss_type = loss_cfg.get("type", "compact_self_supervised")
    stage_weights = loss_cfg.get("stage_weights", DEFAULT_STAGE_WEIGHTS)
    # L_geo (paper lambda_geo = 0.5) needs data.geometry_prior: mono_sparse; gated so
    # Stage 0 can ablate it.
    geo_weight = loss_cfg.get("geo_weight", 0.5) if loss_cfg.get("use_geo_term", False) else 0.0
    components = {}  # per-term losses of the latest step, summed over stages (for logging)

    def stage_loss(pred_depth: torch.Tensor, batch: dict, stage_weight: float) -> torch.Tensor:
        inputs = batch["model_inputs"]
        H, W = pred_depth.shape[-2:]
        scale = W / inputs["ref_img"].shape[-1]

        def resize(img):
            return img if scale == 1.0 else F.interpolate(img, size=(H, W), mode="bilinear", align_corners=False)

        def resize_nearest(x):
            return x if scale == 1.0 else F.interpolate(x, size=(H, W), mode="nearest")

        if loss_type == "compact_self_supervised":
            out = compact_self_supervised_loss(
                pred_depth=pred_depth,
                ref_img=resize(inputs["ref_img"]),
                src_imgs=[resize(s) for s in inputs["src_imgs"]],
                ref_proj=_scale_proj(inputs["ref_proj"], scale),
                src_projs=[_scale_proj(p, scale) for p in inputs["src_projs"]],
                smoothness_weight=loss_cfg.get("smoothness_weight", 0.1),
                photometric_weight=loss_cfg.get("photometric_weight", 1.0),
                prior_depth=resize_nearest(batch["prior_depth"]) if geo_weight > 0 else None,
                prior_valid=resize_nearest(batch["prior_valid"]) if geo_weight > 0 else None,
                geo_weight=geo_weight,
                geo_normalize=loss_cfg.get("geo_normalize", True),
            )
            for key, name in (("photometric_loss", "photo"), ("smoothness_loss", "smooth"), ("geo_loss", "geo")):
                if key in out:
                    components[name] = components.get(name, 0.0) + stage_weight * out[key].item()
            return out["loss"]
        elif loss_type == "supervised_l1":
            gt = F.interpolate(batch["gt_depth"], size=(H, W), mode="nearest")
            mask = F.interpolate(batch["depth_valid_mask"], size=(H, W), mode="nearest")
            return supervised_l1_loss(pred_depth, gt, mask)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def loss_fn(outputs: dict, batch: dict):
        components.clear()
        total = 0.0
        for stage_name, scores in outputs["scores"].items():
            pred_depth = regress_depth(scores, outputs["depth_hypotheses"][stage_name])  # (B, 1, H, W)
            w = stage_weights.get(stage_name, 1.0)
            total = total + w * stage_loss(pred_depth, batch, w)
        return total

    loss_fn.components = components  # read by Trainer for per-term logging
    return loss_fn


CLMVSNET_EPOCHS = 16  # CL-MVSNet's training length; its schedules below are in these epochs


def make_consistency_fn(cfg: dict):
    """CL-MVSNet's ICC + SCC consistency losses (loss.consistency), or None when off.

    Adopted from CL-MVSNet (arXiv 2503.08219) as a baseline correction -- cite it as
    theirs; not part of GeoCorr Lite. See models/losses.py for the mechanism and
    DTUDataset._consistency_inputs for the jittered / resampled views.

    Returns fn(forward, outputs, batch, step) -> generator of weighted losses, one per
    extra forward pass, for Trainer.train_step. The pseudo label is the clean pass's
    (`outputs`) final-stage depth, detached, and supervises every stage of each pass
    with loss.stage_weights.

    Schedules, as in CL-MVSNet but over training progress (step / max_steps mapped to
    its 16 epochs; `e` = that epoch index):
      ICC weight: icc_weight_start doubled at e = 1, 3, 5, 7, 9, capped at icc_weight
        (theirs: 0.01 -> 0.32, adjust_w_icc).
      ICC source dropout: each source element zeroed with p = icc_src_drop_p * e / 15
        (theirs: min(0.1 * epoch / 15, 0.1)). The reference gets one random
        (H // 3) x (W // 3) zero box per batch, excluded from the ICC loss.
      SCC weight: constant scc_weight (theirs 0.01), on pixels whose clean-pass
        final-stage confidence > scc_conf (theirs 0.95).

    Weight scale: CL-MVSNet's unsupervised loss (8 x L0.5 + 6 x SSIM + 0.18 x
    smoothness) is ~61x ours (L_photo + 0.1 L_smooth) on the same depths: ratio
    0.0163, measured on 24 DTU train samples with A0_finetune_clmvs. The defaults are
    CL-MVSNet's weights times 0.0163.
    """
    ccfg = cfg.get("loss", {}).get("consistency", {})
    if not ccfg.get("enabled", False):
        return None
    stage_weights = cfg["loss"].get("stage_weights", DEFAULT_STAGE_WEIGHTS)
    max_steps = cfg.get("training", {}).get("max_steps", 20000)
    icc_weight_start = ccfg.get("icc_weight_start", 0.01 * 0.0163)
    icc_weight = ccfg.get("icc_weight", 0.32 * 0.0163)
    icc_src_drop_p = ccfg.get("icc_src_drop_p", 0.1)
    scc_weight = ccfg.get("scc_weight", 0.01 * 0.0163)
    scc_conf = ccfg.get("scc_conf", 0.95)
    components = {}

    def consistency_fn(forward, outputs, batch, step):
        components.clear()
        epoch = min(int(CLMVSNET_EPOCHS * step / max_steps), CLMVSNET_EPOCHS - 1)
        w_icc = min(icc_weight_start * 2 ** sum(epoch >= e for e in (1, 3, 5, 7, 9)), icc_weight)
        drop_p = min(icc_src_drop_p * epoch / 15, icc_src_drop_p)

        last = list(outputs["scores"])[-1]
        with torch.no_grad():
            final_scores = outputs["scores"][last].detach()
            pseudo = regress_depth(final_scores, outputs["depth_hypotheses"][last])
            confident = cascade_photometric_confidence(final_scores) > scc_conf

            icc = batch["icc_inputs"]
            src_imgs = [x * (torch.rand_like(x) >= drop_p) for x in icc["src_imgs"]]
            H, W = icc["ref_img"].shape[-2:]
            bh, bw = H // 3, W // 3
            y0, x0 = np.random.randint(0, H - bh), np.random.randint(0, W - bw)
            filter_mask = torch.ones_like(icc["ref_img"][:, :1])
            filter_mask[..., y0:y0 + bh, x0:x0 + bw] = 0
            ref_img = icc["ref_img"] * filter_mask

        passes = (
            ("icc", w_icc, filter_mask, {"ref_img": ref_img, "src_imgs": src_imgs}),
            ("scc", scc_weight, confident, batch["scc_inputs"]),
        )
        components.update(icc_weight=w_icc, icc_src_drop_p=drop_p, scc_conf_pct=100.0 * confident.float().mean().item())
        for name, weight, mask, overrides in passes:
            if weight <= 0:
                continue
            aux = forward(**overrides)
            total = 0.0
            for stage, scores in aux["scores"].items():
                depth = regress_depth(scores, aux["depth_hypotheses"][stage])
                total = total + stage_weights.get(stage, 1.0) * consistency_depth_loss(depth, pseudo, mask)
            components[name] = total.item()
            yield weight * total

    consistency_fn.components = components  # read by Trainer for per-term logging
    return consistency_fn


def validate(model, dataset, device: str, view_stride: int) -> dict:
    """Per-stage depth error / GT coverage on every `view_stride`-th val sample."""
    was_training = model.training
    model.eval()
    per_sample = [sample_stage_stats(model, dataset[i], device) for i in range(0, len(dataset), view_stride)]
    model.train(was_training)
    return summarize(per_sample)


def main():
    log_git_commit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage_name = cfg.get("stage_name", os.path.splitext(os.path.basename(args.config))[0])

    for seed in args.seed:
        set_seed(seed)
        run_dir = os.path.join(cfg.get("output_root", "runs"), stage_name, f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)

        model = build_model(cfg["model"])
        trainer_cfg = TrainerConfig(
            max_steps=cfg["training"].get("max_steps", 20000),
            log_every=cfg["training"].get("log_every", 50),
            ckpt_every=cfg["training"].get("ckpt_every", 2000),
            lr=cfg["training"].get("lr", 1e-3),
            log_dir=run_dir,
            stages=None,  # every stage: each narrows the next and carries a loss term
            grad_accum_steps=cfg["training"].get("grad_accum_steps", 1),
            warmup_steps=cfg["training"].get("warmup_steps", 0),
            lr_milestones=cfg["training"].get("lr_milestones"),
            lr_gamma=cfg["training"].get("lr_gamma", 0.5),
        )
        trainer = Trainer(
            model,
            trainer_cfg,
            alpha_schedule=AlphaSchedule(**cfg.get("alpha_schedule", {})),
            gate_schedule=GateCeilingSchedule(**cfg.get("gate_schedule", {})),
            prior_dropout=PriorDropoutSchedule(**cfg.get("prior_dropout", {})),
            device=args.device,
        )

        # Fine-tuning: start from another run's weights (fresh optimizer/LR schedule).
        # A "{seed}" placeholder chains per-seed runs (seedN inits from seedN).
        init_from = cfg["training"].get("init_from")
        if init_from:
            init_from = init_from.format(seed=seed)
            state = torch.load(init_from, map_location=args.device)["model"]
            # Parameters whose shape changed with the architecture (e.g. the
            # regularizer's `up` layer under reg_upsample: trilinear) start fresh, as do
            # modules the source run did not have, listed by key prefix in
            # training.init_new_modules (e.g. the zero-initialized GGF residual);
            # anything else missing or unexpected is still an error.
            own = model.state_dict()
            reinit = sorted(k for k, v in state.items() if k in own and own[k].shape != v.shape)
            new_prefixes = tuple(cfg["training"].get("init_new_modules", []))
            new = sorted(k for k in own if k not in state and new_prefixes and k.startswith(new_prefixes))
            result = model.load_state_dict({k: v for k, v in state.items() if k not in reinit}, strict=False)
            if result.unexpected_keys or set(result.missing_keys) != set(reinit) | set(new):
                raise RuntimeError(f"init_from {init_from}: missing {result.missing_keys}, unexpected {result.unexpected_keys}")
            if new:
                print(f"new modules kept at their init: {new}", flush=True)
            print(f"initialized weights from {init_from}" + (f" (re-initialized {reinit})" if reinit else ""), flush=True)

        # Optional validation depth curve (off unless training.val_every > 0).
        val_every = cfg["training"].get("val_every", 0)
        val_dataset = None
        if val_every > 0:
            from data.datasets.dtu import DTUDataset
            val_dataset = DTUDataset(root=cfg["data"]["root"], split="val", cfg=cfg["data"])
        val_stride = cfg["training"].get("val_view_stride", 1)
        val_log = os.path.join(run_dir, "val_depth.jsonl")

        def run_validation(at_step: int):
            t_val = time.time()
            result = validate(trainer.model, val_dataset, args.device, val_stride)
            for stage, s in result.items():
                for k in ("median_err_mm", "coverage_pct", "pct_gt_4mm"):
                    trainer.writer.add_scalar(f"val/{stage}_{k}", s[k], at_step)
            with open(val_log, "a") as f:
                f.write(json.dumps({"step": at_step, **result}) + "\n")
            print(f"[val step {at_step}] " + "  ".join(
                f"{stage}: med {s['median_err_mm']:.2f}mm cov {s['coverage_pct']:.1f}%" for stage, s in result.items()
            ) + f"  ({time.time() - t_val:.0f}s)", flush=True)

        loss_fn = make_loss_fn(cfg)
        consistency_fn = make_consistency_fn(cfg)
        if consistency_fn is not None:
            # the dataset builds the augmented (ICC) and resampled-view (SCC) inputs
            cfg["data"]["consistency"] = True
        train_loader = make_dataloader(cfg, "train")
        accum = trainer_cfg.grad_accum_steps
        step = 0
        if val_dataset is not None:
            run_validation(0)
        t_start = time.time()
        micro_batches = []
        for epoch in range(cfg["training"].get("max_epochs", 10**9)):
            for batch in train_loader:
                micro_batches.append(move_batch_to_device(batch, args.device))
                if len(micro_batches) < accum:
                    continue
                stats = trainer.train_step(
                    micro_batches if accum > 1 else micro_batches[0], step, loss_fn=loss_fn, consistency_fn=consistency_fn
                )
                micro_batches = []
                if step % trainer_cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / (step + 1) * (trainer_cfg.max_steps - step - 1)
                    print(f"[step {step}/{trainer_cfg.max_steps} epoch {epoch}] loss {stats['loss']:.4f} "
                          f"lr {stats['lr']:.2e} {stats['step_time']:.2f}s/step "
                          f"elapsed {elapsed / 3600:.2f}h eta {eta / 3600:.2f}h", flush=True)
                if step % trainer_cfg.ckpt_every == 0:
                    trainer.save_checkpoint(os.path.join(run_dir, f"ckpt_{step}.pt"), step)
                step += 1
                if val_dataset is not None and step % val_every == 0:
                    run_validation(step)
                if step >= trainer_cfg.max_steps:
                    break
            if step >= trainer_cfg.max_steps:
                break

        trainer.save_checkpoint(os.path.join(run_dir, "ckpt_final.pt"), step)
        if val_dataset is not None and step % val_every != 0:
            run_validation(step)
        print(f"[{stage_name} seed={seed}] training complete -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
