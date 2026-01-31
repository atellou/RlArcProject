# Experiment Observations (Rewards → Curriculum → Entropy → CARSM)

This document consolidates the qualitative observations so far across the experiment YAMLs we discussed/edited:

- `experiments/configs/reward_exp.yaml`
- `experiments/configs/curriculum_exp.yaml`
- `experiments/configs/entropy_exp.yaml`
- `experiments/configs/base.yaml` (defaults were updated during the process)

The central recurrent phenomenon across runs is a **collapse into a high `no_op` region** (policy becomes dominated by actions that do not change the grid / effectively stop acting), often paired with:

- learning to adjust **submission frequency** (sometimes the main thing that improves)
- limited/unclear learning of meaningful pixel edits (improvement signal often weak)

Throughout this document, “no-op region/collapse” refers to the empirical behavior you observed: once the policy starts choosing no-ops frequently, entropy tends to drop and the system quickly becomes stuck.

> **Important note on evidence:** You explicitly noted you don’t have consistent numeric summaries for all experiments (and not all were run to identical horizons). This review therefore focuses on the qualitative/relative behaviors you described (e.g., “enters no-op faster/slower”, “learns to not submit”, “flat metrics”, etc.).

---

## 1) Reward Sweep (First Iteration) — `reward_exp.yaml`

### Overall observation

- **Reward design mattered**, but many reward configurations still allowed collapse or produced flat learning.
- A key theme: shaping and penalties interact with the emergent no-op behavior.
- One candidate emerged as the most promising base reward: **`ex13_all_penalties_plus_positive_shaping`**, which later became the “ex13 reward base” you built curriculum/entropy work on.

### Per-experiment summary (from config intent + what we later observed)

#### `ex1_no_penalty`

- **Intent:** no explicit discouragement of wrong submits, no-ops, or stalling.
- **Expected behavior:** agent can spam submits and/or make meaningless edits with little downside.
- **Observed later pattern (general):** such weak constraints tend to produce unstable training signals and degenerate policies.

#### `ex2_wrong_submit_only`

- **Intent:** discourage wrong submissions, but no explicit no-op penalty.
- **Expected:** may reduce submitting, but can still get stuck doing nothing if value estimates favor safety.

#### `ex3_no_op_only` (`zero_improvement_penalty=0.25`)

- **Intent:** push the agent away from no-op via explicit penalty.
- **Observed theme later:** strong anti-no-op pressure can sometimes “mask” no-op collapse early (agent keeps changing pixels), but not necessarily create meaningful improvement learning.

#### `ex4_step_cost_only` (`step_cost=0.01`)

- **Intent:** time pressure; encourages solving quickly and not taking meaningless steps.
- **Observed theme later:** step cost can interact with no-op/stalling and submission behavior.

#### `ex5_wrong_submit_plus_no_op`

- Combined constraints; intended to reduce wrong submits and discourage no-ops.

#### `ex6_wrong_submit_plus_step_cost`

- Adds time pressure to wrong-submit penalty.

#### `ex7_perfect_no_submit_only` (`perfect_no_submit_penalty_coef=0.50`)

- **Intent:** prevent the “perfect but never submit” local optimum.

#### `ex8_wrong_submit_plus_no_op_plus_step_cost`

- “More realistic” shaping by combining wrong submit, no-op discouragement, and step cost.

#### `ex9_wrong_submit_plus_no_op_plus_perfect_no_submit`

- Adds “submit when solved” incentive to the above.

#### `ex10_all_penalties_plus_negative_shaping` (`shaping_clamp_min=-0.25`)

- **Intent:** allow negative shaping signal.
- **Observed theme later:** negative shaping interacting with no-op collapse was risky because “not acting” can become comparatively safe.

#### `ex11_all_penalties_plus_positive_shaping` (`shaping_clamp_min=0.25`)

- **Intent:** enforce positive-only shaping.

#### `ex12_all_penalties_plus_positive_shaping` (higher `zero_improvement_penalty=0.75`)

- **Intent:** stronger pressure against stagnation.

#### `ex13_all_penalties_plus_positive_shaping` (the “ex13 reward base”)

- **Config:**

  - `wrong_submit_penalty=1.0`
  - `step_cost=0.01`
  - `perfect_no_submit_penalty_coef=0.50`
  - `zero_improvement_penalty=0.1`
  - `shaping_clamp_min=0.5`
- **Selection rationale (as you described earlier):**

  - “Works best” relative to others.
  - More robust behavior vs collapse compared to other reward designs.
  - Became the foundation for curriculum and later entropy/CARSM exploration.

---

## 2) Curriculum Experiments — `curriculum_exp.yaml`

### Shared setup

All curriculum experiments in this file use the **ex13 reward base** (same env reward knobs) and vary:

- curriculum enable/disable
- curriculum replacement probabilities:

  - `on_warmup`
  - `on_episode_start` for `episode_start_steps`
  - `on_train`

You also experimented with curriculum oracle action strategy probabilities in code (`ActionStrategy` in `rlarcworld/environments/wrappers/curriculum.py`), but the core finding remained: **more curriculum often accelerated no-op collapse**.

### Core qualitative findings you reported

- **No curriculum**:

  - “Never enters no-op” for a long time
  - but also “never improves” / metrics stay flat
- **More aggressive curriculum**:

  - tends to enter no-op sooner
  - can learn submission timing better
  - can increase pixel-wise reward but still be in a no-op-dominated regime

This strongly suggests curriculum replacement was acting like a “crutch” that improved some short-horizon behavior (submission) while encouraging a conservative/stalling policy.

### Per-experiment summary

#### `ex13_base_no_curriculum`

- **Key result:**
  - did **not** collapse into no-op (at least within the horizon you tested)
  - but also did **not** learn (flat metrics)
- **Interpretation:** exploration exists but is not converted into consistent improvement gradients.

#### `ex13_curriculum_mild`

- Mild ongoing curriculum (`episode_start=0.25`, `on_train=0.05`).
- **Observed trend:** curriculum tends to accelerate entering no-op vs no-curriculum.

#### `ex13_curriculum_medium`

- Medium ongoing curriculum (`episode_start=0.50`, `on_train=0.10`).
- **Observed trend:** stronger curriculum generally increased the speed of collapse.

#### `ex13_curriculum_aggressive`

- Aggressive ongoing curriculum (`episode_start=1.0`, `on_train=0.25`).
- **Observed trend:** faster entry into no-op; but may learn some “when to submit” behavior.

#### `ex13_curriculum_mild_entropy_floor`

- This became important later as the “best delay” configuration.
- Adds:

  - `entropy_floor_coef=0.02`
  - `min_entropy=0.2`
- **Observed:**

  - took longer to enter no-op
  - but “never learned to not submit so often” (submission behavior stayed high)

#### `ex13_curriculum_mild_entropy_floor_heads`

- Derived to address frequent submits:

  - `submit_entropy_scale=0.25`
  - `color_entropy_scale=1.25`
- Intended to reduce submit randomness and increase color exploration.

#### `ex14_curriculum_only_start_up`

- “Curriculum only at startup”:

  - `on_episode_start=1.0` for first 50 steps
  - `on_train=0.0`
- Goal: bootstrap early without permanent crutch.
- **Outcome (overall curriculum conclusion):** still not a reliable fix; curriculum family considered “not working” as a whole.

#### `ex15_curriculum_medium_50_50`

- Ongoing curriculum with moderate on-train (`on_train=0.1`) and a different oracle action strategy (you annotated 50/50 improve/degrade).
- **General observed trend:** didn’t resolve the core collapse behavior.

#### `ex16_curriculum_only_improve`

- Oracle biased to always improve (per your later edits/notes).
- **General observed trend:** still did not solve collapse; curriculum generally judged unsuccessful.

### Curriculum “Trigger” attempt (code-level)

You experimented with a proposed “replace only when predicted no-op” trigger (and asked to wire it behind a flag). Empirically:

- **Result:** it entered the no-op state even faster.
- **Conclusion:** predicted-no-op gating is not worth pursuing further.

---

## 3) Entropy Regularization Experiments — `entropy_exp.yaml`

### Overall observation you reported

Across the entropy experiments:

- Most runs ended up with **high `no_op` (~100% by the end)**.
- The differences were mostly:
  - how quickly they learn to reduce submit frequency
  - how long before they collapse into no-op

Also:

- `entropy_floor_base_plus_small_entropy_coef` was slower to learn “don’t submit frequently”, and also the last to enter no-op — consistent with **slower learning** rather than fundamentally better learning.
- `entropy_floor_weak` kept submission rate low (didn’t “move the submit term”), but still ended high no-op.

### Current experiment keys in file

#### `no_entropy_regularization`

- Control.
- **Observed overall:** collapse still occurs; does not provide stable meaningful signal.

#### `entropy_floor_weak` (`entropy_floor_coef=0.01`, `min_entropy=0.1`)

- **Observed:**
  - kept submit rate low (submission term stays near zero)
  - still ended with high no-op

#### `entropy_floor_base` (`0.02`, `0.2`)

- **Observed overall:** reasonable baseline; still collapses.

#### `entropy_floor_strong` (`0.05`, `0.3`)

- **Observed:** did not end at 100% no-op (relative best among these), but still did not yield a strong learning signal.

#### `entropy_floor_base_plus_small_entropy_coef`

- Adds `entropy_coef=0.01` (decay 0.99) on top of base floor.
- **Observed:** slowest to learn “don’t submit often”, last to enter no-op — likely due to slower learning.

#### `ex16_entropy_floor_base_heads`

- Adds per-head scaling:

  - `submit_entropy_scale=0.25`
  - `color_entropy_scale=1.25`
- **Observed:** you later noted that head scaling can help the agent learn to not always submit and can delay no-op entry.

#### `ex13_curriculum_mild_entropy_floor_with_heads_entropy`

- You created this as a “best so far” combined baseline:

  - ex13 reward base
  - mild curriculum schedule
  - entropy floor base (`0.02`, `0.2`)
  - head entropy scaling:

    - `submit_entropy_scale=0.25`
    - `color_entropy_scale=1.5`
- **Observed (your statement):**

  - still no meaningful reward signal overall
  - but it does two valuable things:

    - takes considerable time to enter no-op
    - learns to not submit so frequently
  - “closest to learning something that we have”

---

## 4) CARSM Next Step — `entropy_exp.yaml`

### `ex18_add_carsm`

- This is the CARSM experiment you defined as:
  - Base: `ex13_curriculum_mild_entropy_floor_with_heads_entropy`
  - plus `d4pg.carsm: true`

Rationale (your conclusion):

- Even though the reward/improvement signal is weak, this baseline produces the **best behavior so far** in terms of:
  - delaying no-op collapse
  - learning “not always submit”

This is a reasonable baseline to test CARSM because CARSM may improve representation/credit assignment under sparse/weak signals.

---

## 5) Base Configuration Changes — `base.yaml`

During iteration, `experiments/configs/base.yaml` was updated to align defaults with the selected direction:

### Env defaults shifted toward the ex13 reward base

- `step_cost: 0.01`
- `perfect_no_submit_penalty_coef: 0.50`
- `zero_improvement_penalty: 0.1`
- `shaping_clamp_min: 0.5`

### Training curriculum defaults set to “mild”

- `enable_curriculum: true`
- `on_episode_start: 0.25`
- `on_train: 0.05`

### D4PG entropy floor default applied

- `entropy_floor_coef: 0.02`
- `min_entropy: 0.2`

This means many “default” runs now implicitly assume the ex13-ish regime unless overridden.

---

## 6) Cross-cutting Conclusions (as of now)

### Curriculum (as implemented) is not a solution

- Aggressive curriculum consistently pushed the system toward no-op collapse sooner.
- Even trigger-based gating (predicted no-op) worsened collapse.
- Best we got from curriculum was “delay collapse a bit” in combination with entropy floor, but not a real learning breakthrough.

### Entropy floor helps stability but does not create a strong learning signal

- Floors can delay collapse.
- Adding a small entropy coefficient can slow learning (and sometimes delay collapse), but doesn’t necessarily improve outcome.

### Per-head entropy scaling is the most promising entropy tweak so far

- Reducing submit entropy while increasing color entropy appears to:
  - reduce “always submit” behavior
  - delay collapse

### Selection objective should shift

Earlier selection used “time-to-no-op” as a primary metric.

Now, since many variants just trade off learning speed vs collapse time, a better selection objective is:

- the best **trajectory quality** prior to collapse:
  - reduced pathological submit
  - meaningful action diversity
  - any measurable improvement trend

Given your latest assessment, `ex13_curriculum_mild_entropy_floor_with_heads_entropy` is the best available candidate for that.

---

## 7) Suggested Next Measurements (to make future comparisons less ambiguous)

If you add/confirm these logs, future selections will be less subjective:

- `submit_rate` over time
- `no_op_rate` over time
- mean `improvement` (from env info) over time
- reward breakdown (`pixel_wise`, `binary`, submit term if logged)
- entropy breakdown per head:
  - submit entropy
  - color entropy
  - location entropy

---

## 8) Current Best Candidate to Move Forward

Based on your final stated criterion (“closest to learning something”) and observed behavior:

- **Best baseline to test CARSM:** `ex13_curriculum_mild_entropy_floor_with_heads_entropy`
- **CARSM test run:** `ex18_add_carsm`

---
