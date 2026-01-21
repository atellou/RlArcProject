# Reward experiments notes

## Context

These notes summarize observations from the reward experiments run so far (2 epochs, ~3000 iterations/epoch) and serve as a running log to guide the next round of sweeps.

## Reward experiments: First iteration

- **no_submit_penalty**
  - **Entropy**: varies and remains relatively low (reported ~0.2 to ~0.8) throughout training.
  - **Gradients**: remain active (reported actor/critic gradient norms varying roughly 0 to 4).
  - **Reward**: continues to vary; training does not immediately collapse.

- **added_submit_penalty**
  - **Entropy**: settles early (reported ~4000 iterations).
  - **Gradients**: settle / die early (reported ~4000 iterations).
  - **Reward**: mostly ~0 with very small variations (~1e-3), suggesting the agent may not perceive a strong learning signal.

- **added_improvement_penalty**
  - **Behavior**: very similar to `added_submit_penalty`.
  - **Entropy/gradients**: settle / die early.
  - **Reward**: mostly ~0 with very small variations (~1e-3).

- **all_penalties**
  - **Entropy**: grows high (reported ~5 by end of training).
  - **Actor gradients**: die early (reported ~2000 iterations).
  - **Critic gradients**: remain active (reported ~0.3 to 1).
  - **Reward**: cyclical; overall mostly negative.

## Reward experiments: Second iteration

Second iteration variants were created to reduce constant negative drift (lower `zero_improvement_penalty`) and reduce the perfect-no-submit penalty.

- **all_penalties_low_improvement_penalty** (`perfect_no_submit_penalty_coef=0.75`, `zero_improvement_penalty=0.05`)
  - **Outcome**: similar to first iteration overall.
  - **Observation**: gradients still die; entropy becomes extremely high.

- **all_penalties_low_perfect_no_submit_penalty** (`perfect_no_submit_penalty_coef=0.25`, `zero_improvement_penalty=0.25`)
  - **Outcome**: similar to first iteration overall.
  - **Observation**: gradients still die; entropy becomes extremely high.

- **all_penalties_low_both** (`perfect_no_submit_penalty_coef=0.25`, `zero_improvement_penalty=0.05`)
  - **Outcome**: the only second-iteration variant with sustained movement.
  - **Entropy**: shows variation until the end.
  - **Gradients**: continue to vary until the end (no early death compared to the other two variants).

## Global observations so far

- **Rewards are mostly negative**, with positive rewards being rare.
- For multiple penalty configurations, training tends to fall into regimes where:
  - **Actor gradients die early** while the critic keeps moving (or entropy increases sharply).
  - **Rewards hover near 0** or are cyclical, suggesting weak/ambiguous policy gradient signal.

## Hypotheses / next actions

- **Signal-to-noise**: penalties may dominate shaping early, producing a “do-nothing / don’t-submit” attractor or a noisy high-entropy regime.
- **Next candidate base reward config** (from reward sweeps only):
  - `all_penalties_low_both` is currently the best candidate for continued experimentation because it avoids early actor gradient death.
- **Next sweeps to consider on top of `all_penalties_low_both`**:
  - Entropy regime selection (entropy_coef / floor / decay) and then submit-head entropy scaling.
  - A small LR sanity sweep if gradients are still unstable.
