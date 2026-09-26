"""CL-MVSNet consistency terms (models/losses.py, training/train.py::make_consistency_fn)."""
import numpy as np
import torch
import torch.nn.functional as F

from models.losses import cascade_photometric_confidence, consistency_depth_loss
from training.train import make_consistency_fn


def _clmvsnet_confidence(score):
    """Verbatim from CL-MVSNet networks/clmvsnet.py RegressionDepth.forward."""
    prob_volume = F.softmax(score, dim=1)
    num_depth = prob_volume.shape[1]
    prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1,
                                        padding=0).squeeze(1)
    hyp = torch.arange(num_depth, dtype=torch.float).view(1, -1, 1, 1)
    depth_index = torch.sum(prob_volume * hyp, 1).long().clamp(min=0, max=num_depth - 1)
    return torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)


def test_confidence_matches_clmvsnet():
    torch.manual_seed(0)
    score = torch.randn(2, 8, 5, 6) * 3
    ours = cascade_photometric_confidence(score)
    assert ours.shape == (2, 1, 5, 6)
    assert torch.allclose(ours[:, 0], _clmvsnet_confidence(score), atol=1e-6)


def test_confidence_extremes():
    peaked = torch.full((1, 8, 1, 1), -50.0)
    peaked[0, 3] = 50.0
    assert cascade_photometric_confidence(peaked).item() > 0.999
    # uniform: expected index 3.5 -> floor 3 -> bins 2..5 -> 4/8
    assert abs(cascade_photometric_confidence(torch.zeros(1, 8, 1, 1)).item() - 0.5) < 1e-6


def test_consistency_loss_masks_and_downsamples():
    pseudo = torch.full((1, 1, 8, 8), 500.0)
    mask = torch.ones(1, 1, 8, 8)
    mask[..., :4, :] = 0  # top half unsupervised
    pred = torch.full((1, 1, 4, 4), 500.0, requires_grad=True)
    with torch.no_grad():
        pred[..., :2, :] = 900.0  # wrong only where masked out
    loss = consistency_depth_loss(pred, pseudo, mask)
    assert loss.item() == 0.0
    loss.backward()  # still differentiable
    empty = consistency_depth_loss(pred, pseudo, torch.zeros_like(mask))
    assert empty.item() == 0.0 and empty.requires_grad


def test_make_consistency_fn_off_by_default():
    assert make_consistency_fn({"loss": {"type": "compact_self_supervised"}}) is None
    assert make_consistency_fn({"loss": {"consistency": {"enabled": False}}}) is None


def test_consistency_fn_passes_and_pseudo_label():
    cfg = {"loss": {"consistency": {"enabled": True, "icc_weight_start": 2.0, "icc_weight": 2.0,
                                    "scc_weight": 3.0, "scc_conf": 0.95}}}
    fn = make_consistency_fn(cfg)
    D = 8
    hyp = {"coarse": torch.linspace(500, 570, D).view(1, D, 1, 1).expand(1, D, 2, 2),
           "fine": torch.linspace(500, 507, D).view(1, D, 1, 1).expand(1, D, 4, 4)}
    confident = torch.full((1, D, 4, 4), -50.0)
    confident[:, 2] = 50.0  # clean pass: fine depth 502, confident everywhere
    clean = {"scores": {"coarse": torch.zeros(1, D, 2, 2), "fine": confident}, "depth_hypotheses": hyp}
    calls = []

    def forward(**overrides):
        calls.append(sorted(overrides))
        return {"scores": {"coarse": torch.zeros(1, D, 2, 2, requires_grad=True),
                           "fine": torch.zeros(1, D, 4, 4, requires_grad=True)},
                "depth_hypotheses": hyp}

    batch = {"icc_inputs": {"ref_img": torch.ones(1, 3, 4, 4), "src_imgs": [torch.ones(1, 3, 4, 4)]},
             "scc_inputs": {"src_imgs": 0, "src_projs": 0}}
    losses = list(fn(forward, clean, batch, step=0))
    assert calls == [["ref_img", "src_imgs"], ["src_imgs", "src_projs"]]
    # uniform aux outputs: coarse depth 535, fine 503.5; pseudo label 502 everywhere
    per_pass = 0.5 * (33 - 0.5) + 2.0 * (1.5 - 0.5)  # smooth L1, stage weights coarse 0.5 / fine 2
    assert np.allclose([l.item() for l in losses], [2.0 * per_pass, 3.0 * per_pass], rtol=1e-4)
    assert fn.components["scc_conf_pct"] == 100.0


def test_consistency_schedules_follow_clmvsnet():
    """w_icc 0.01 doubled at CL-MVSNet epochs 1/3/5/7/9 (cap 0.32); p_icc = 0.1 * epoch / 15;
    here over 1600 steps = 16 epochs of 100 steps."""
    cfg = {"loss": {"consistency": {"enabled": True, "icc_weight_start": 0.01, "icc_weight": 0.32,
                                    "icc_src_drop_p": 0.1, "scc_weight": 0.0}},
           "training": {"max_steps": 1600}}
    fn = make_consistency_fn(cfg)
    D = 4
    hyp = {"fine": torch.linspace(500, 503, D).view(1, D, 1, 1).expand(1, D, 6, 6)}
    clean = {"scores": {"fine": torch.zeros(1, D, 6, 6)}, "depth_hypotheses": hyp}
    seen = {}

    def forward(**overrides):
        seen["ref_img"] = overrides["ref_img"]
        return {"scores": {"fine": torch.zeros(1, D, 6, 6, requires_grad=True)}, "depth_hypotheses": hyp}

    batch = {"icc_inputs": {"ref_img": torch.ones(1, 3, 6, 6), "src_imgs": [torch.ones(1, 3, 6, 6)]}, "scc_inputs": {}}
    expected = {0: (0.01, 0.0), 100: (0.02, 0.1 / 15), 350: (0.04, 0.3 / 15), 950: (0.32, 0.9 / 15), 1599: (0.32, 0.1)}
    for step, (w, p) in expected.items():
        assert len(list(fn(forward, clean, batch, step))) == 1  # SCC skipped at weight 0
        assert np.isclose(fn.components["icc_weight"], w) and np.isclose(fn.components["icc_src_drop_p"], p)
        assert (seen["ref_img"] == 0).sum().item() == 3 * 2 * 2  # one (H//3, W//3) box zeroed
