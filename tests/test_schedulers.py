from rlarcworld.utils.schedulers import BetaScheduler


def test_beta_scheduler_monotonic_and_clamp():
    s = BetaScheduler(start=0.4, end=1.0, steps=5)
    vals = [s.beta]
    for _ in range(10):
        vals.append(s.step())
    # Monotonic non-decreasing
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    # Final value should not exceed end (allow tiny numerical tolerance)
    assert vals[-1] <= 1.0 + 1e-6


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
