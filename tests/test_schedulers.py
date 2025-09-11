import pytest

from rlarcworld.utils.schedulers import BetaScheduler


@pytest.mark.parametrize(
    "start,end,steps",
    [
        (0.0, 1.0, 1),
        (0.1, 1.0, 5),
        (0.4, 1.0, 5),
        (-0.5, 0.8, 4),  # start below 0 is allowed
        (0.2, 2.0, 4),  # end above 1 is allowed
        (0.9, 0.3, 3),  # decreasing schedule (now fixed)
        (1.5, -0.2, 4),  # decreasing schedule with negative end
    ],
)
def test_beta_scheduler_monotonic_and_clamp(start, end, steps):
    s = BetaScheduler(start=start, end=end, steps=steps)
    vals = [s.beta]
    for _ in range(steps * 2):  # step beyond schedule to verify saturation
        vals.append(s.step())

    # Monotonic movement towards end (allow equal due to saturation)
    if end >= start:
        assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    else:
        assert all(b <= a for a, b in zip(vals, vals[1:])), vals

    # Should reach the target within the scheduled steps
    assert abs(vals[steps] - end) <= 1e-6, (start, end, steps, vals[: steps + 1])

    # Should saturate at end value after steps (allowing for floating point precision)
    assert abs(vals[-1] - end) <= 1e-6, vals


def test_beta_scheduler_state_dict_roundtrip():
    s = BetaScheduler(start=0.1, end=0.9, steps=3)
    s.step()
    s.step()
    state = s.state_dict()

    s2 = BetaScheduler(start=0.0, end=0.0, steps=1)
    s2.load(state)

    assert s2.start == 0.1
    assert s2.end == 0.9
    assert s2.steps == 3
    assert s2.internal_step == s.internal_step
    assert s2.beta == s.beta
