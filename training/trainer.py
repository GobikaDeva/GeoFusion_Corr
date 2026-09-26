"""Training loop with the safeguards from docs/baseline_recovery_plan.md.

Responsibilities of this trainer that go beyond a generic training loop:
    - Applies the alpha / gate-ceiling schedules (training/schedule.py) each step.
    - Applies geometry-prior dropout + noise on a configured fraction of samples.
    - Logs gate mean/std separately for textured / textureless / reflective /
      invalid-prior pixels (requires region masks -- see evaluation/hard_region_masks.py).
    - Records wall-clock time, peak GPU memory, and parameter count for the
      efficiency gate (docs/evaluation_and_gates.md).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from .schedule import AlphaSchedule, GateCeilingSchedule, PriorDropoutSchedule


@dataclass
class TrainerConfig:
    max_steps: int = 20000
    log_every: int = 50
    ckpt_every: int = 2000
    lr: float = 1e-3
    log_dir: str = "runs/exp"
    # Cascade stages to compute per forward pass. Stages are computed in order up
    # to the deepest one listed, since each narrows the next (see
    # models.geofusionnet.GeoFusionNet.forward). None means every stage.
    stages: Optional[List[str]] = None
    # Micro-batches per optimizer step (effective batch = batch_size * grad_accum_steps).
    grad_accum_steps: int = 1
    # Linear LR warmup over the first `warmup_steps` optimizer steps, then the LR is
    # multiplied by `lr_gamma` at each fraction of max_steps in `lr_milestones`
    # (e.g. [0.6, 0.75, 0.9]). Defaults leave the LR constant.
    warmup_steps: int = 0
    lr_milestones: Optional[List[float]] = None
    lr_gamma: float = 0.5


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: TrainerConfig,
        alpha_schedule: Optional[AlphaSchedule] = None,
        gate_schedule: Optional[GateCeilingSchedule] = None,
        prior_dropout: Optional[PriorDropoutSchedule] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.alpha_schedule = alpha_schedule or AlphaSchedule()
        self.gate_schedule = gate_schedule or GateCeilingSchedule()
        self.prior_dropout = prior_dropout or PriorDropoutSchedule()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.writer = SummaryWriter(cfg.log_dir)
        self.param_count = sum(p.numel() for p in model.parameters())

    def lr_at(self, step: int) -> float:
        lr = self.cfg.lr
        if self.cfg.warmup_steps > 0 and step < self.cfg.warmup_steps:
            lr *= (step + 1) / self.cfg.warmup_steps
        for frac in self.cfg.lr_milestones or []:
            if step >= int(round(frac * self.cfg.max_steps)):
                lr *= self.cfg.lr_gamma
        return lr

    def _log_gate_by_region(self, step: int, gate: torch.Tensor, region_masks: dict):
        """region_masks: {"textured": bool mask, "textureless": ..., "reflective": ...,
        "invalid_prior": ...}, each broadcastable to `gate`'s shape.
        Guards against gate collapse (docs/risks_and_mitigations.md)."""
        for region, mask in region_masks.items():
            if mask.sum() == 0:
                continue
            vals = gate[mask]
            self.writer.add_scalar(f"gate/{region}_mean", vals.mean().item(), step)
            self.writer.add_scalar(f"gate/{region}_std", vals.std().item(), step)

    def train_step(self, batch, step: int, loss_fn, region_masks_fn=None, consistency_fn=None) -> dict:
        """One optimizer step. `batch` is a batch dict, or a list of
        `grad_accum_steps` micro-batch dicts whose gradients are accumulated.

        `consistency_fn(forward, outputs, batch, step)` (optional, see
        training/train.py::make_consistency_fn) yields extra losses, each from its own
        forward pass; each is backpropagated as soon as it is yielded so only one
        pass's graph is alive at a time. Its pseudo labels are detached, so the
        gradient equals that of the summed loss."""
        self.model.train()
        alpha = self.alpha_schedule.value(step)
        max_gate = self.gate_schedule.value(step)
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        micro_batches = batch if isinstance(batch, list) else [batch]
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for batch in micro_batches:
            def forward(**overrides):
                return self.model(**{**batch["model_inputs"], **overrides}, alpha=alpha, max_gate=max_gate, stages=self.cfg.stages)

            outputs = forward()
            loss = loss_fn(outputs, batch) / len(micro_batches)
            loss.backward()
            total += loss.item()
            if consistency_fn is not None:
                for aux_loss in consistency_fn(forward, outputs, batch, step):
                    aux_loss = aux_loss / len(micro_batches)
                    aux_loss.backward()
                    total += aux_loss.item()
        self.optimizer.step()
        step_time = time.time() - t0

        if step % self.cfg.log_every == 0:
            self.writer.add_scalar("train/loss", total, step)
            for fn in (loss_fn, consistency_fn):
                for name, value in getattr(fn, "components", {}).items():
                    self.writer.add_scalar(f"train/{name}", value, step)
            self.writer.add_scalar("schedule/alpha", alpha, step)
            self.writer.add_scalar("schedule/max_gate", max_gate, step)
            self.writer.add_scalar("schedule/lr", lr, step)
            self.writer.add_scalar("perf/step_time_sec", step_time, step)
            if torch.cuda.is_available():
                self.writer.add_scalar(
                    "perf/peak_mem_mb", torch.cuda.max_memory_allocated() / 1e6, step
                )
            if outputs.get("geocorr") and region_masks_fn is not None:
                for stage_name, gc_out in outputs["geocorr"].items():
                    self._log_gate_by_region(step, gc_out["gate"], region_masks_fn(batch, stage_name))

        return {"loss": total, "lr": lr, "alpha": alpha, "max_gate": max_gate, "step_time": step_time}

    def save_checkpoint(self, path: str, step: int):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "param_count": self.param_count,
        }, path)
