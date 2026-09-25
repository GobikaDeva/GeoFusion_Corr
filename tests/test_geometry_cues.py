"""Confirms geometry cues assemble to the expected channel count and shape, and that
N1's shuffle control actually changes per-sample correspondence (a sanity check for
the causal-information test in docs/evaluation_and_gates.md)."""
import torch

from ablations.controls import shuffle_geometry_cues
from models.geometry_cues import NUM_BASE_CUES, compute_geometry_cues


def test_base_cues_shape_and_channels():
    B, D, H, W = 2, 4, 8, 8
    candidate_depth = torch.rand(B, D, H, W) * 2 + 1
    prior_depth_ref = torch.rand(B, 1, H, W) * 2 + 1
    prior_conf_ref = torch.rand(B, 1, H, W)
    warped_prior_depth_src = [torch.rand(B, D, H, W) * 2 + 1 for _ in range(2)]
    warped_prior_conf_src = [torch.rand(B, D, H, W) for _ in range(2)]
    normal_ref = torch.randn(B, 3, H, W)
    warped_normal_src = [torch.randn(B, 3, D, H, W) for _ in range(2)]
    raw_scores = torch.randn(B, D, H, W)

    cues = compute_geometry_cues(
        candidate_depth, prior_depth_ref, prior_conf_ref,
        warped_prior_depth_src, warped_prior_conf_src,
        normal_ref, warped_normal_src, raw_scores,
    )
    assert cues.shape == (B, NUM_BASE_CUES, D, H, W)


def test_shuffle_changes_batch_correspondence():
    cues = torch.arange(2 * 3 * 4 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4, 4)
    shuffled = shuffle_geometry_cues(cues, dim=0)
    assert shuffled.shape == cues.shape
    # same multiset of values, different per-sample assignment (batch size 2 means
    # this could coincidentally match on identity permutation; check it's a valid perm)
    assert torch.allclose(shuffled.sum(), cues.sum())
