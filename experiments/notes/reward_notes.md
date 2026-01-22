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
  - **Entropy**: grows high (reported ~5 by end of training).
  - **Gradients**: actor gradients die after ~1000 steps.
  - **Critic gradients**: settle around ~0.2 to ~0.4, but with more scale variation than actor.
  - **Loss**:
    - Actor loss shows cyclic variation with a slow downward trend; lowest observed ~-0.8 (slightly stronger trend than `all_penalties_low_perfect_no_submit_penalty`).
    - Critic loss oscillates ~1.0 to ~1.5.
  - **Reward dynamics**:
    - Cyclic / noisy cosine-like pattern.
    - Binary reward roughly ~-0.25 to ~-0.6.
    - Pixel-wise reward roughly ~-0.8 to ~-1.25.

- **all_penalties_low_perfect_no_submit_penalty** (`perfect_no_submit_penalty_coef=0.25`, `zero_improvement_penalty=0.25`)
  - **Outcome**: similar to first iteration overall.
  - **Entropy**: grows high (reported ~5 by end of training).
  - **Gradients**: actor gradients die after ~1000 steps.
  - **Critic gradients**: settle around ~0.2 to ~0.4, but with more scale variation than actor.
  - **Loss**:
    - Actor loss shows cyclic variation with a slow downward trend; lowest observed ~-0.8.
    - Critic loss oscillates ~1.0 to ~1.5.
  - **Reward dynamics**:
    - Cyclic / noisy cosine-like pattern.
    - Binary reward roughly ~-0.25 to ~-0.6.
    - Pixel-wise reward roughly ~-0.8 to ~-1.25.

- **all_penalties_low_both** (`perfect_no_submit_penalty_coef=0.25`, `zero_improvement_penalty=0.05`)
  - **Outcome**: the only second-iteration variant with sustained movement.
  - **Entropy**: shows drastic variation until the end.
  - **Gradients**:
    - More variation (reported ~0 to ~2) than the other two second-iteration variants.
    - Both actor and critic show periods close to zero and periods with clear movement (intermittent learning).
  - **Loss**:
    - Actor loss shows erratic changes; lowest observed ~-2.
    - Critic loss shows erratic changes; lowest observed ~0.08.
  - **Reward dynamics**:
    - Long plateaus with abrupt/erratic changes.
    - Plateaus often occur at values ending in ~.05.
    - Both binary and pixel-wise rewards remain mostly negative, varying roughly ~0.05 to ~-1.05.

## Global observations so far

- **Rewards are mostly negative**, with positive rewards being rare.
- For multiple penalty configurations, training tends to fall into regimes where:
  - **Actor gradients die early** while the critic keeps moving (or entropy increases sharply).
  - **Rewards hover near 0** or are cyclical, suggesting weak/ambiguous policy gradient signal.
- **Replay / behavior confound (important for curriculum runs)**: outcome metrics reflect the behavior policy. When curriculum is enabled, compare metrics split by `curriculum/used` (actor-only vs curriculum-driven) rather than comparing raw reward curves across runs.
- **Gradient collapse signature**: repeated pattern is critic continues fitting while actor gradients collapse (often coincident with high entropy or flat rewards). This is the primary failure mode observed so far.
- **Plateau / quantization in rewards**: long plateaus at discrete reward values suggest the reward is effectively piecewise-constant for most transitions (sparse improvement). This motivates curriculum as a way to increase the frequency of informative transitions early.

## Hypotheses / next actions

- **Signal-to-noise**: penalties may dominate shaping early, producing a “do-nothing / don’t-submit” attractor or a noisy high-entropy regime.
- **Next candidate base reward config** (from reward sweeps only):
  - `all_penalties_low_both` is currently the best candidate for continued experimentation because it avoids early actor gradient death.
- **Next sweeps to consider on top of `all_penalties_low_both`**:
  - Entropy regime selection (entropy_coef / floor / decay) and then submit-head entropy scaling.
  - A small LR sanity sweep if gradients are still unstable.

- **Reward configs selected for curriculum tests (next iteration)**:
  - Keep `all_penalties_low_both`:
    - Only second-iteration reward variant with sustained/intermittent actor+critic gradient movement.
    - Best candidate for curriculum to improve data distribution and potentially turn intermittent learning into consistent progress.
  - Keep `all_penalties_low_improvement_penalty`:
    - Serves as a stress-test reward configuration where the actor tends to die early without curriculum.
    - Useful to test whether curriculum can prevent early collapse by providing better early replay signal.
  - Discard for now: `all_penalties_low_perfect_no_submit_penalty`:
    - Behaves very similarly to `all_penalties_low_improvement_penalty` (early actor gradient death + cyclic rewards).
    - Running both under curriculum is likely redundant; keep one “dying gradient” config as a stress test.

- **Curriculum motivation**:
  - None of the above reward-only runs produced success or even near-success behavior.
  - Next experiments should compare reward settings under curriculum schedules.

- **Curriculum comparisons to run**:
  - Warm-start schedule (oracle-heavy at episode start) to reduce early signal starvation.
  - Ongoing-assist schedule (small `on_train` probability) to prevent collapse/stagnation.
