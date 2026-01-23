# Reward experiments notes

## Context

These notes summarize observations from recent reward sweeps and serve as a running log to guide the next round of experiments.

## Latest sweep summary

- **Submission success rate:** `0` across the sweep (no successful submissions observed).
- **Grid perfect rate:** never reaches a perfect grid state.
- **Episode length:** episodes consistently run until `max_steps_per_episode`.
- **Curriculum usage in training batches:** decays to ~0 after a few hundred steps when curriculum is warmup-only.

Interpretation:

- With warmup-only curriculum, oracle transitions become a shrinking fraction of the replay buffer as new policy transitions are added, so `curriculum_used` in sampled batches naturally trends toward 0.
- Given `submit_success_rate == 0` and no perfect grids, reward shaping alone is currently not producing a solvable learning signal within the training budget.

## Reward config observations (high level)

- **Actor entropy (averaged over heads):**
  - `no_submit_penalty` produced the highest actor entropy throughout the run.
    - Interpretation: removing the perfect-no-submit penalty reduces pressure to commit to a deterministic submit policy, so the learned policy remains more stochastic for longer.
  - `all_penalties_low_both` was consistently the next-highest.
    - Interpretation: lower penalties (especially the lower perfect-no-submit penalty) appear to preserve exploration relative to the higher-penalty regimes.
  - Most other reward configurations fell below ~`0.5` by ~iteration `2800` (total training horizon was `6000` iterations).
    - Interpretation: these configurations tend to converge toward more deterministic behavior by mid-training, which can reflect either stable learning or early convergence to a suboptimal attractor (given zero success, the latter is plausible).

- **Critic distribution entropy (per head):**
  - **Pixel-wise head:**
    - `no_submit_penalty` and `added_submit_penalty` were the highest and stabilized around ~`0.9`.
      - Interpretation: the pixel-wise value distribution remains broad/uncertain, suggesting the critic does not collapse to a narrow confident prediction for pixel edits under these reward regimes.
    - `all_penalties_low_both` and `all_penalties_low_improvement_penalty` were typically around ~`0.8` with visibly higher variance.
      - Interpretation: these regimes show similar uncertainty but with more non-stationarity, consistent with ongoing changes in the policy’s edit distribution.
  - **Binary head:**
    - `all_penalties` and `all_penalties_medium_improvement_penalty` were low (around ~`0.1`).
      - Interpretation: the critic becomes highly confident for the binary outcome under these penalties; without any observed submit success, this likely reflects confident prediction of failure/negative outcomes.
    - `all_penalties_low_perfect_no_submit_penalty` was intermediate (around ~`0.5`).
      - Interpretation: reducing perfect-no-submit penalty appears to increase uncertainty in the binary outcome, consistent with more varied submission behavior.
    - All other reward configurations were high (around ~`0.9`).
      - Interpretation: the binary outcome remains diffuse/uncertain, suggesting the agent does not settle into a consistent submit/no-submit regime that yields predictable outcomes.

- **Gradient norms (actor and critic):**
  - Gradients remained non-zero to the end of training.
    - Interpretation: optimization did not fully stagnate; learning signals persisted even late in training.
  - Magnitude was generally low-to-moderate (reported overall range ~`0.1` to `2`).
    - Interpretation: updates are present but small, consistent with incremental policy changes rather than abrupt shifts.
  - Several configurations showed persistent late-stage variability (reported fluctuations >~`0.1` continuing through the end), consistent with non-stationary learning dynamics rather than a flatline collapse.
    - Interpretation: the system continues to explore or oscillate among behaviors, which matches the observed non-stationary rewards and lack of convergence to success.

- **Rewards (component vs total):**
  - Per-component reward curves (when separated by shaping components) generally converged toward the expected penalty-dominated regimes.
    - Interpretation: the shaping terms dominate the reward landscape, and the agent’s behavior does not escape the regime where penalties determine the mean return.
  - In contrast, the **total reward** and **binary reward** curves for:
    - `all_penalties_low_both`
    - `added_improvement_penalty`
    - `all_penalties_low_improvement_penalty`
    remained visibly non-stationary (did not settle by the end of the `6000` iterations).
    - Interpretation: these configurations maintain ongoing behavioral change (or oscillation) in aspects that affect end-of-episode outcomes (binary rewards), even though this does not translate into successful submissions.

- **Losses:**
  - `all_penalties` and `all_penalties_medium_improvement_penalty` became nearly stationary after ~iteration `3000`.
    - Interpretation: these regimes converge to a stable critic/policy update pattern relatively early; given zero success, this stability may correspond to convergence to a failure mode.
  - `added_submit_penalty` showed a critic loss that decreased smoothly (described as logarithmically approaching 0) through the end.
    - Interpretation: this reward regime may produce a more consistent TD target distribution (easier to fit), but reduced critic loss alone does not imply improved task performance.
  - Several other configurations remained erratic, with critic loss varying roughly between `0.04` and `0.6` through late training.
    - Interpretation: these regimes maintain non-stationary TD targets and/or unstable value estimates, consistent with ongoing exploration or policy oscillation.
  - Actor loss:
    - `all_penalties_low_perfect_no_submit_penalty` and `all_penalties_medium_improvement_penalty` were closest to `0` compared to other configs.
      - Interpretation: the policy gradient signal becomes small under these reward settings, which can indicate either near-convergence or weak advantage estimates; given zero success, weak/degenerate learning is plausible.
    - `no_submit_penalty` and `added_submit_penalty` plateaued around the `-1.5` to `-2` range for most of training.
      - Interpretation: these configurations reach a stable policy gradient regime that does not further improve; combined with zero success, this suggests convergence to a non-solving behavior.

## Action metrics that matter most right now

- Because submit success is 0 and perfect grids are never reached, the priority for the next iteration is to increase the frequency of *informative* transitions rather than fine-tuning penalty magnitudes.

- Configs with **non-trivial submit-rate dynamics** (submit rate not stuck at exactly 0 or 1):
  - `all_penalties_low_both`
  - `all_penalties_low_improvement_penalty`

## Next iteration recommendation

Given the above, prefer **ongoing curriculum influence (B)** rather than warmup-only.

Suggested levers:

- Increase replay buffer capacity so early oracle transitions persist longer.
- Add a small `on_train` curriculum rate (e.g. 1-5%) so the replay distribution continues to include solvable/improving trajectories.
- Keep episode-start curriculum off (avoid permanently masking the learned policy and avoid the "always curriculum" trap when episodes reset frequently).

## Decision-making: minimal next experiments

Goal: determine whether curriculum can produce any non-zero success signal (submission success or perfect grids) with the smallest possible experiment set.

### Reward configurations to carry forward (2)

- `all_penalties_low_both` (`perfect_no_submit_penalty_coef=0.25`, `zero_improvement_penalty=0.05`)
  - Interpretation: chosen because it exhibited non-trivial submission-rate dynamics and sustained non-stationary learning signals.

- `all_penalties_low_improvement_penalty` (`perfect_no_submit_penalty_coef=0.75`, `zero_improvement_penalty=0.05`)
  - Interpretation: chosen as a contrasting penalty regime that still showed non-trivial submission-rate dynamics.

### Curriculum schedules to compare (2)

- **Ongoing-only** (`on_warmup=1.0`, `on_episode_start=0.0`, `on_train=0.02`, `episode_start_steps=0`)
  - Interpretation: isolates whether a small continual trickle of curriculum transitions is enough to shape the replay distribution without masking episode-start behavior.

- **Episode-start + ongoing** (`on_warmup=1.0`, `on_episode_start=0.5`, `episode_start_steps=50`, `on_train=0.02`)
  - Interpretation: provides stronger guidance early in each episode. If this still yields zero success, the bottleneck is likely not simply exploration/signal starvation.

This yields **4 total runs** (2 reward configs × 2 curriculum schedules).

### Stop criteria / decision rules

Evaluate after ~`1000` to `2000` iterations (no need to wait to `6000` for a go/no-go decision):

- **Breakthrough condition:** any of the following becomes non-zero:
  - submission success rate
  - perfect grid rate
  - improvement rate (if tracked) trending upward in actor-only samples
  - Interpretation: proceed to longer runs and only then consider additional sweeps (entropy, LR, etc.).

- **Failure condition:** if all 4 runs keep:
  - submission success rate == 0
  - perfect grid rate == 0
  - episodes always terminate at max steps
  - Interpretation: deprioritize further reward-coefficient sweeps and focus instead on action-space / submit mechanics / curriculum strength or training objective changes.
