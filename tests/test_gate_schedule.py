"""Confirms the alpha and gate-ceiling schedules ramp correctly, per
docs/baseline_recovery_plan.md ("residual scale at zero... warm up... ramp the
maximum geometry gate from approximately 0.05 to 0.10")."""
from training.schedule import AlphaSchedule, GateCeilingSchedule


def test_alpha_schedule_zero_during_warmup():
    sched = AlphaSchedule(warmup_steps=100, ramp_steps=100, alpha_max=1.0)
    assert sched.value(0) == 0.0
    assert sched.value(99) == 0.0


def test_alpha_schedule_ramps_to_max():
    sched = AlphaSchedule(warmup_steps=100, ramp_steps=100, alpha_max=1.0)
    assert sched.value(100) == 0.0
    assert 0.0 < sched.value(150) < 1.0
    assert sched.value(200) == 1.0
    assert sched.value(1000) == 1.0  # clamps at max, doesn't overshoot


def test_gate_ceiling_within_recommended_range():
    sched = GateCeilingSchedule(warmup_steps=0, ramp_steps=100, gate_max=0.10)
    final = sched.value(1000)
    assert 0.05 <= final <= 0.10 or final == 0.10
    assert sched.value(0) == 0.0
