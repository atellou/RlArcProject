# RlArcWorld Neural Network Architectures

This document describes the neural network architectures defined in this repository for reinforcement learning on the ARC (Abstraction and Reasoning Corpus) dataset. The architectures build upon the ViTARC encoder (arXiv:2410.06405) and adapt it for actor-critic RL.

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RlArcWorldViTARC                                  │
│  (Shared Encoder - nn_modules.py)                                           │
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────┐    │
│  │ GridEncoder │───▶│ CustomT5Stack│───▶│ ExampleCrossAttention       │    │
│  │             │    │ (ViTARC)     │    │ (optional, for examples)    │    │
│  └─────────────┘    └──────────────┘    └─────────────────────────────┘    │
│         │                                           │                       │
│         ▼                                           ▼                       │
│  ViTARCEncoderOutput: {hidden_states, attention_mask, ...}                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ ActionHeadFrom    │   │ CriticHeadFrom    │   │ RewardHeadFrom    │
│ Encoder           │   │ Encoder           │   │ Encoder           │
│ (actor.py)        │   │ (critic.py)       │   │ (reward_world_    │
│                   │   │                   │   │  model.py)        │
└───────────────────┘   └───────────────────┘   └───────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   Action Probs           Q-Distributions         Reward Predictions
   (x, y, color,          (per reward type)       (binary, pixel_wise)
    submit)
```

Notes:

- The encoder is implemented in `RlArcWorldViTARC` (`nn_modules.py`) and uses `GridEncoder` (`grid_encoder.py`) to convert grids into token sequences.
- Object-based positional encoding (OPE) uses OpenCV contour detection (`ObjectIndexExtractor`).
- The critic head keys are driven by the runtime `distributional` configuration (e.g. `binary`, `pixel_wise`, and optionally `*_n_reward`).

## Core Components

### 1. `RlArcWorldViTARC` (nn_modules.py)

The shared encoder module that processes ARC grid states. This is the foundation used by all downstream heads.

**Key Adaptations:**

- **Direct Token Encoding**: Grid color values (0-10) are used directly as token IDs—no text tokenizer needed
- **Temporal Grid Support**: Encodes sequences of grids (initial → last → current) with learned temporal embeddings
- **Example Conditioning**: Cross-attention mechanism allows the encoder to attend to input-output example pairs

**Configuration (`RlArcWorldViTARCConfig`):**

```python
@dataclass
class RlArcWorldViTARCConfig:
    grid_size: int           # Grid dimensions (H=W)
    d_model: int = 128       # Hidden dimension
    d_ff: int = 512          # Feed-forward dimension
    num_layers: int = 4      # Transformer layers
    num_heads: int = 8       # Attention heads
    dropout: float = 0.1
    ape_type: str | None     # Absolute position encoding type
    rpe_type: str | None     # Relative position encoding type
    use_ope: bool | None     # Object-based positional encoding
    ape_mixer: str | None    # Position embedding mixer
    use_index_grid: bool | None # Index/timestamp grid encoding
    use_temporal: bool | None # Temporal grid encoding
    use_examples: bool | None # Example grid encoding
```

**Default Best Configuration** (from ViTARC paper ablation):

- `ape_type`: `"SinusoidalAPE2D"` — 2D sinusoidal absolute position encoding
- `rpe_type`: `"Two-slope-Alibi"` — Largest performance gain
- `ape_mixer`: `"weighted_sum_no_norm_vec"` — Second largest gain
- `use_ope`: `True` — Third largest gain (object-based encoding)

**Output (`ViTARCEncoderOutput`):**

```python
@dataclass
class ViTARCEncoderOutput:
    hidden_states: torch.Tensor      # [B, seq_len, d_model]
    attention_mask: torch.Tensor     # [B, seq_len]
    encoder_outputs: BaseModelOutput # Full transformer outputs
    input_ids: torch.Tensor          # Token IDs
    temporal_ids: torch.Tensor       # Temporal position IDs (optional)
    object_idx: torch.Tensor         # Object indices for OPE (optional)
    example_hidden: torch.Tensor     # Example encodings (optional)
    example_mask: torch.Tensor       # Example attention mask (optional)
```

---

### 2. `GridEncoder` (grid_encoder.py)

Converts ARC grids into token sequences for the transformer.

**Key Features:**

- **Raster-order Tokenization**: Flattens 2D grids with row/column separators
- **Object Index Extraction**: Uses OpenCV contour detection to identify objects for OPE
- **Special Tokens**: BOS, EOS, SEP, PAD, ROW_SEP, COL_SEP tokens
- **Temporal Grid Encoding**: Concatenates multiple grids with separators
- **Example Pair Encoding**: Encodes input-output pairs for cross-attention

**Implementation Notes:**

- OPE object indices are produced by `ObjectIndexExtractor` and returned as integer tensors (not uint8). This avoids overflow once the object counter exceeds 255.
- The OpenCV path uses a simple bounding-box fill for each contour (as in the ViTARC paper).

---

### 3. `ActionHeadFromEncoder` (nn_modules.py)

Converts encoder hidden states into action probability distributions.

**Architecture:**

```txt
Encoder Hidden States [B, seq_len, H]
           │
           ▼
┌─────────────────────────────────────┐
│  Learnable Query (per action type)  │  ← 4 queries: x, y, color, submit
│  [1, 1, H] → expand → [B, 1, H]     │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Cross-Attention                    │
│  Q: action query                    │
│  K,V: encoder hidden states         │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  LayerNorm + Linear Classifier      │
│  → Softmax → Action Probabilities   │
└─────────────────────────────────────┘
```

**Output:** `TensorDict` with softmax distributions:

- `x_location`: [B, grid_size]
- `y_location`: [B, grid_size]
- `color_values`: [B, num_colors]
- `submit`: [B, 2]

---

### 4. `CriticHeadFromEncoder` (nn_modules.py)

Converts encoder hidden states + action distributions into distributional Q-values (D4PG-style).

**Architecture:**

```txt
Encoder Hidden States [B, seq_len, H]    Action Distributions
           │                                     │
           │                    ┌────────────────┘
           │                    ▼
           │         ┌─────────────────────────┐
           │         │ Concat + Linear Project │
           │         │ → Action Token [B, 1, H]│
           │         └─────────────────────────┘
           │                    │
           ▼                    ▼
┌─────────────────────────────────────────────┐
│  Concatenate: [hidden_states, action_token] │
│  → [B, seq_len + 1, H]                      │
└─────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Learnable Query (per reward type)  │  ← e.g., binary, pixel_wise
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Cross-Attention + LayerNorm        │
│  + Linear → Softmax over atoms      │
└─────────────────────────────────────┘
```

**Output:** `TensorDict` with Q-value distributions.

The set of reward heads is determined by the keys present in the `distributional` configuration passed to `VitARCCritic` (`num_atoms`, `v_min`, `v_max`). In particular, when n-step returns are enabled, the critic is expected to also include:

- `pixel_wise_n_reward`
- `binary_n_reward`

so that critic outputs match the reward dict produced by the steppers.

- `binary`: [B, num_atoms_binary] — probability distribution over value atoms
- `pixel_wise`: [B, num_atoms_pixel] — probability distribution over value atoms

---

### 5. `RewardHeadFromEncoder` (nn_modules.py)

Predicts rewards directly from encoder hidden states (for world model).

**Architecture:** Similar to `ActionHeadFromEncoder` but without action conditioning:

- **Binary Reward**: Softmax classification [B, 2] for (incorrect, correct)
- **Pixel-wise Reward**: Sigmoid regression [B] for percentage of wrong pixels

---

### 6. `ExampleCrossAttention` (nn_modules.py)

Multi-layer cross-attention module for conditioning on example pairs.

**Purpose:** Allows the grid representation to attend to encoded example pairs, learning the transformation pattern demonstrated by the examples.

**Architecture:**

- Multiple cross-attention layers (grid queries → example keys/values)
- Layer normalization after each attention
- Feed-forward network with GELU activation

---

## High-Level Model Classes

### `ViTARCActor` (actor.py)

Complete actor network for policy learning.

```python
class ViTARCActor(nn.Module):
    def __init__(self, color_values, rl_vit_arc_config, config=None):
        self.rl_vit_arc = RlArcWorldViTARC(rl_vit_arc_config)
        self.action_heads = ActionHeadFromEncoder(...)

    def forward(self, state: TensorDict) -> TensorDict:
        encoder_output = self.rl_vit_arc(state)
        action_logits = self.action_heads(encoder_output.hidden_states, ...)
        return TensorDict({k: softmax(v) for k, v in action_logits.items()})
```

### `VitARCCritic` (critic.py)

Complete critic network for distributional value estimation.

```python
class VitARCCritic(nn.Module):
    def __init__(self, color_values, rl_vit_arc_config, num_atoms, v_min, v_max, ...):
        self.rl_vit_arc = RlArcWorldViTARC(rl_vit_arc_config)
        self.critic_head = CriticHeadFromEncoder(...)
        self.z_atoms = {...}  # Value support for each reward type

    def forward(self, state: TensorDict, action: TensorDict) -> TensorDict:
        encoder_output = self.rl_vit_arc(state)
        q_distributions = self.critic_head(encoder_output, action)
        return q_distributions
```

### `ViTARCRewardModel` (reward_world_model.py)

Reward prediction model for model-based RL / world model training.

```python
class ViTARCRewardModel(nn.Module):
    def __init__(self, color_values, rl_vit_arc_config, config=None):
        self.rl_vit_arc = RlArcWorldViTARC(rl_vit_arc_config)
        self.reward_head = RewardHeadFromEncoder(...)

    def forward(self, state: TensorDict) -> TensorDict:
        encoder_output = self.rl_vit_arc(state)
        return self.reward_head(encoder_output)
```

---

## Training/Runtime Integration Notes

- Rewards are produced by steppers (e.g. `PixelWiseArcStepper`) and are converted into a dict/`TensorDict` with base keys (`pixel_wise`, `binary`) and, when `n_steps > 1`, additional n-step keys (`pixel_wise_n_reward`, `binary_n_reward`).
- To keep training consistent, both training and validation steppers should be constructed with the same `n_step_config` so reward keys match the critic head keys.
- `NStepMinVMaxConfig` bounds can be derived from the `distributional` configuration to ensure consistent clipping and value supports.

---

## Design Principles

1. **Shared Encoder**: All models share the same `RlArcWorldViTARC` encoder architecture, enabling:
   - Consistent state representations across actor/critic/reward model
   - Potential for weight sharing or transfer learning
   - Unified configuration via `RlArcWorldViTARCConfig`

2. **Learnable Query Attention**: Task-specific heads use learnable queries that attend to encoder outputs, allowing the model to extract relevant information for each task (actions, Q-values, rewards).

3. **Modular Head Design**: Each head (`ActionHeadFromEncoder`, `CriticHeadFromEncoder`, `RewardHeadFromEncoder`) is a standalone module that can be:
   - Tested independently
   - Swapped or extended without modifying the encoder
   - Configured with different hyperparameters

4. **TensorDict Integration**: All inputs/outputs use `TensorDict` for:
   - Type-safe key access via enums (`StateKeys`, `ActionKeys`, `RewardKeys`)
   - Automatic batching and device management
   - Consistent interface across the codebase

---

## File Structure

```txt
rlarcworld/models/
├── nn_modules.py          # Core building blocks
│   ├── RlArcWorldViTARCConfig
│   ├── ViTARCEncoderOutput
│   ├── RlArcWorldViTARC
│   ├── ExampleCrossAttention
│   ├── ActionHeadFromEncoder
│   ├── CriticHeadFromEncoder
│   └── RewardHeadFromEncoder
├── grid_encoder.py        # Grid tokenization and object detection
├── actor.py               # ViTARCActor
├── critic.py              # VitARCCritic
├── reward_world_model.py  # ViTARCRewardModel
└── vit_arc.py             # Base ViTARC transformer (CustomT5Stack)
```
