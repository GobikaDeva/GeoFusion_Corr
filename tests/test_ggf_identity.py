"""B1 (configs/ablations/B1_ggf_input.yaml) adds the zero-initialized GGF residual to
A0f. Loaded with A0f's weights and before any training, its scores must be
bit-identical to A0f's at every cascade stage, whatever the geometry input."""
import torch

from models import build_model
from training.train import load_config


def _inputs(B=1, H=64, W=80):
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])

    def proj(tx):
        P = torch.eye(4)
        E = torch.eye(4)
        E[0, 3] = tx
        P[:3, :4] = K @ E[:3, :4]
        return P[None].expand(B, -1, -1)

    return dict(
        ref_img=torch.randn(B, 3, H, W),
        src_imgs=[torch.randn(B, 3, H, W) for _ in range(4)],
        ref_geom=torch.randn(B, 4, H, W),
        ref_proj=proj(0.0),
        src_projs=[proj(t) for t in (-20.0, -10.0, 10.0, 20.0)],
        depth_hypotheses_per_stage={"coarse": torch.linspace(425, 935, 48)[None].expand(B, -1)},
        depth_interval=torch.full((B,), 2.65),
    )


def test_b1_matches_a0f_at_init():
    torch.manual_seed(0)
    a0f = build_model(load_config("configs/ablations/A0f_fixes_only_short.yaml")["model"]).eval()
    b1 = build_model(load_config("configs/ablations/B1_ggf_input.yaml")["model"]).eval()
    result = b1.load_state_dict(a0f.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(k.startswith("backbone.ggf_fusion.") for k in result.missing_keys)

    inputs = _inputs()
    with torch.no_grad():
        ref = a0f(**inputs)["scores"]
        out = b1(**inputs)["scores"]
    assert set(out) == {"coarse", "mid", "fine"}
    for stage in out:
        assert torch.equal(out[stage], ref[stage]), stage
