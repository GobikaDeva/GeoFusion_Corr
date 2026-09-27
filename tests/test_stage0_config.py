"""configs/stage0_baseline.yaml is the long-form copy of the frozen Stage 0 config (A0f)."""
from training.train import load_config


def test_stage0_baseline_matches_a0f():
    assert load_config("configs/stage0_baseline.yaml") == load_config("configs/ablations/A0f_fixes_only_short.yaml")
