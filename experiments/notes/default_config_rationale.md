# Default config rationale (baseline)

This note documents the rationale for the default values in `experiments/configs/base.yaml`.

The goal of these defaults is to provide a *representative, stability-first baseline* for subsequent reward / entropy / n-step sweeps.

## Invariants and key checks

- The environment produces multiple reward heads:
  - `pixel_wise`, `binary` (1-step)
  - `pixel_wise_n_reward`, `binary_n_reward` (n-step)

- The code asserts that the distributional critic heads exactly match the reward keys in the batch:
  - `rlarcworld/algorithms/d4pg.py` enforces
    `set(next_q_response.keys()) == set(reward.keys())`.

- When `n_steps > 1`, training requires distributional config entries for the n-step keys:
  - `rlarcworld/trainers/arc_d4pg.py` validates that `num_atoms[...]` exists for
    `pixel_wise_n_reward` and `binary_n_reward`.

These checks make it safe to run a single baseline with `n_steps=1` while still logging both 1-step and n-step rewards, and to run explicit n-step sweeps with separate distributional supports keyed by `*_n_reward`.

## Environment defaults (`env`)

- **`color_values: 11`**

  The training pipeline uses padding/sentinel values in addition to the 10 ARC colors.
  Using `11` avoids action validity mismatches and keeps actor/critic head sizes aligned.

- **`perfect_no_submit_penalty_coef: 0.75`**

  This makes "perfect but not submitted" undesirable, while staying less severe than a
  hard failure. This is intended to prevent policies from stalling in a nearly-correct
  state without over-incentivizing premature submission.

- **`zero_improvement_penalty: 0.25`**

  A small penalty for no-improvement steps discourages inaction and repeated edits that
  do not change the grid. This helps reduce stagnant trajectories.

## D4PG defaults (`d4pg`)

- **`n_steps: 1`**

  TD(0) is used as the default baseline because it is the easiest regime to debug and
  interpret, and is less sensitive to return-range / distributional support mismatch.
  N-step variants should be studied explicitly via sweeps (e.g. LR × n-steps experiments).

- **Entropy regularization (stability-first)**

  The actor entropy has shown both runaway growth and collapse across runs.
  The baseline reduces entropy pressure while retaining some exploration:

  - `entropy_coef: 0.01`
  - `entropy_floor_coef: 0.0`
  - `min_entropy: 0.2`
  - `submit_entropy_scale: 0.25`

  Rationale:
  - Reducing the global entropy coefficient avoids persistent randomness that can mask
    learning signal and cause oscillatory behavior.
  - Lowering submit entropy specifically helps reduce erratic submission behavior while
    keeping exploration available for the edit heads.

## Distributional defaults (`distributional`)

- **Atoms:** `21` (baseline resolution)
- **Support:** `v_min=-2`, `v_max=1`

The return support is widened on the negative side to account for accumulated penalties
(e.g. no-improvement penalty) and avoid saturating the distributional projection at
`v_min`, which can lead to dead gradients.

For n-step experiments, the `*_n_reward` heads should use more negative `v_min` values
appropriate to the selected `n_steps`.

## Replay buffer defaults (`replay_buffer`)

- **PER parameters:** `alpha=0.2`, `beta=0.2`

This keeps prioritized replay mild. More aggressive PER can amplify bias when early
experience is dominated by negative rewards.

## Curriculum defaults (`training`)

- **`enable_curriculum: true` (warmup-only)**

The baseline enables curriculum only during warmup as a pragmatic "prefill-like" strategy
to seed the replay buffer with higher-quality transitions before learning begins.

The default schedule sets `on_episode_start=0.0` and `on_train=0.0` so that once warmup
ends, collected experience reflects the learned policy.

Warmup curriculum action selection uses `ArcActions.ActionStrategy` defaults:

- `improve: 0.7`
- `no_change: 0.1`
- `no_change_submit: 0.05`
- `degrade: 0.1`
- `degrade_submit: 0.05`
- `on_perfect_match_submit: 0.9`

These defaults bias the warmup buffer toward improvement trajectories while still
including some no-op and degrade cases for diversity.

Determinism: with fixed seeds (`reproducibility.*_seed`) and deterministic torch mode,
the warmup curriculum is reproducible run-to-run because the strategy and location
sampling use torch RNG.
