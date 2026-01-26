# Curriculum experiments notes

## Context

These notes summarize observations from curriculum schedule experiments (ongoing vs episode-start vs top-left curriculum) under recent stabilized training settings.

Experiments referenced:

- `low_both_curriculum_ongoing`
- `low_both_curriculum_episode_start_ongoing`
- `low_improvement_penalty_curriculum_ongoing`
- `low_both_top_left_curriculum_permanent`
- `low_improvement_penalty_top_left_curriculum_permanent`

## Summary of outcomes

### Learning signal / reward shaping

- **Most curriculum variants** produce very similar reward shaping behavior, the values stay stable around **`-0.05`**, the penalty introduced for no improvement.
- **Shaping mean** for the majority of runs settles around **`-0.05`**.
  - Interpretation: the policy likely converges to (or remains near) a **safe / low-change regime**, without learning consistently correct edits.
- **Stronger curriculum (top_left permanent)** diverges:
  - **Shaping mean** settles around **`~-0.25`** on average.
  - **High noise**, mostly negative.
  - Interpretation: the policy is at least **not stuck in the “no change” attractor**, but behavior is unstable and overall performance is worse for pixel-wise returns.

### Pixel-wise and binary rewards

- **Pixel-wise**:
  - Strong top-left curriculum remains **worse** than the mid curriculum strategies.
  - Mid/ongoing curricula perform better relative to top-left but still do not reach positive shaping.
- **Binary**:
  - Overall similar across runs, generally around **`-1`**.

### Losses

- **Critic loss**:
  - Top-left curricula show **higher critic loss** (1.5 avg) than other curricula (0.5 avg).
  - `low_both_curriculum_ongoing` have the **lowest critic losses** (0.2 avg).
- **Actor loss**:
  - Roughly similar across experiments.
  - Tends to plateau around **`~-2`**.
  - Interpretation: policy gradient updates are happening but appear to converge to a similar regime across curricula.

### Entropy

- **Policy (actor) entropy** is noisy across all experiments.
- The **ongoing-only** curricula (no episode-start curriculum) have **lower average entropy** than the others.
  - Interpretation: ongoing-only may push earlier toward more deterministic behavior (not necessarily better behavior).

### Gradient norms

- **Critic gradients**:
  - **Top-left curriculum** runs maintain higher critic gradient norms, roughly **~`3`**.
  - Other curricula are generally **below `1`** on average.
  - Interpretation: top-left curricula drive larger/less stable TD updates.
- **Actor gradients**:
  - `low_both_top_left_curriculum_permanent` is highest, roughly **`1.0`–`1.4`**.
  - Other runs average around **`0.2`–`0.9`**.
  - `low_both_curriculum_ongoing` trends closest to **`0`**.

## Notable event: submit behavior change (~iteration 3500)

Observed for `low_improvement_penalty_curriculum_ongoing`:

- Around **iteration ~`3500`**:
  - **Submission failure rate drops**.
  - **Submission rate remains stable**.
  - Interpretation: the agent learned **when to submit** (or at least improved the submit/no-submit decision boundary), not by submitting more often but by submitting more appropriately.
- Despite that:
  - **Shaping reward remains ~unchanged**.
  - Best improvement described:
    - Binary policy moves from **`-1`** to (unclear exact value; described as an improvement)
    - Pixel-wise from **`-1`** to **`0`**
- **Submission reward term mean** moves toward **`0`** by the end of training.
  - Interpretation: submit decision quality improves, but edit quality still does not translate into consistent task solving.

## Interpretation / takeaways

- Across all schedules tested so far, **no curriculum strategy reliably yields positive shaping reward**.
- The overall pattern suggests:
  - Either **more training** is required for the policy to escape the penalty-dominated regime, or
  - Reward shaping/curriculum alone is insufficient; the agent may need **stronger incentives**, better exploration support, or changes in objective/credit assignment.
- Stronger curriculum (top-left permanent) does induce different (more active) behavior, but appears to:
  - Increase critic instability (higher loss / higher grad norms)
  - Worsen pixel-wise outcomes relative to mid curricula

## Next actions (hypotheses)

- Compare curricula under identical optimizer settings and seeds (already being done).
- Consider testing one or more of:
  - Stronger positive incentives for correct pixel edits
  - Mechanisms to increase density of informative transitions (e.g., replay/curriculum sampling schedule)
  - Additional submit-specific shaping (given the observed submit improvement event)
