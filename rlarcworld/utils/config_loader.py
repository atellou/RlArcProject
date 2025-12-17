"""Configuration loader utilities for experiment management.

Provides functions to load YAML configs, merge with base configs,
and convert to ArcD4PGConfig dataclass instances.
"""

from pathlib import Path

import yaml

from rlarcworld.common.configs import (
    ArcD4PGConfig,
    CheckpointConfig,
    D4PGConfig,
    DistributionalConfig,
    EnvConfig,
    NStepMinVMaxConfig,
    ReplayBufferConfig,
    RlArcWorldViTARCConfig,
    SystemConfig,
    TrainingConfig,
    WorldModelConfig,
)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict.

    Args:
        base: Base dictionary to merge into.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary with override values taking precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed YAML as dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment_config(
    experiment_name: str,
    experiments_path: str | Path = "experiments/configs/experiments.yaml",
    base_path: str | Path = "experiments/configs/base.yaml",
) -> dict:
    """Load an experiment config merged with base config.

    Args:
        experiment_name: Name of the experiment in experiments.yaml.
        experiments_path: Path to experiments YAML file.
        base_path: Path to base config YAML file.

    Returns:
        Merged configuration dictionary.

    Raises:
        KeyError: If experiment_name not found in experiments file.
    """
    base_config = load_yaml(base_path)
    experiments = load_yaml(experiments_path)

    if experiment_name not in experiments:
        available = list(experiments.keys())
        raise KeyError(
            f"Experiment '{experiment_name}' not found. Available: {available}"
        )

    experiment_overrides = experiments[experiment_name]
    # Remove description field (not part of config)
    experiment_overrides.pop("description", None)

    return deep_merge(base_config, experiment_overrides)


def dict_to_config(config_dict: dict) -> ArcD4PGConfig:
    """Convert a configuration dictionary to ArcD4PGConfig.

    Args:
        config_dict: Configuration dictionary (e.g., from YAML).

    Returns:
        ArcD4PGConfig instance.
    """
    # Build n_step_config if present
    d4pg_section = config_dict.get("d4pg", {})
    n_step_dict = d4pg_section.get("n_step_config", {})
    n_step_config = NStepMinVMaxConfig(**n_step_dict) if n_step_dict else None

    # Build env config first so we can reuse grid_size
    env_section = config_dict.get("env", {})
    env_config = EnvConfig(**env_section)

    # Build network config, auto-filling grid_size from env if missing
    network_dict = config_dict.get("network", {}).copy()
    if network_dict and "grid_size" not in network_dict:
        network_dict["grid_size"] = env_config.grid_size
    network_config = RlArcWorldViTARCConfig(**network_dict)

    # Build D4PG config with optional n_step_config
    d4pg_dict = d4pg_section.copy()
    # Be robust to YAML strings for numeric fields (e.g. "3e-4")
    float_fields = [
        "policy_lr",
        "critic_lr",
        "gamma",
        "tau",
        "entropy_coef",
        "entropy_coef_decay",
        "max_grad_norm",
    ]
    for key in float_fields:
        if key in d4pg_dict and d4pg_dict[key] is not None:
            d4pg_dict[key] = float(d4pg_dict[key])
    if n_step_config:
        d4pg_dict["n_step_config"] = n_step_config
    elif "n_step_config" in d4pg_dict:
        del d4pg_dict["n_step_config"]
    d4pg_config = D4PGConfig(**d4pg_dict)

    dist_dict = config_dict.get("distributional", {})
    distributional_config = DistributionalConfig(**dist_dict)

    replay_dict = config_dict.get("replay_buffer", {})
    replay_config = ReplayBufferConfig(**replay_dict)

    training_dict = config_dict.get("training", {})
    training_config = TrainingConfig(**training_dict)

    world_model_dict = config_dict.get("world_model", {})
    world_model_config = WorldModelConfig(**world_model_dict)

    checkpoint_dict = config_dict.get("checkpoint", {})
    checkpoint_config = CheckpointConfig(**checkpoint_dict)

    system_dict = config_dict.get("system", {})
    system_config = SystemConfig(**system_dict)

    return ArcD4PGConfig(
        dataset=config_dict.get("dataset", ""),
        env=env_config,
        network=network_config,
        d4pg=d4pg_config,
        distributional=distributional_config,
        replay_buffer=replay_config,
        training=training_config,
        world_model=world_model_config,
        checkpoint=checkpoint_config,
        system=system_config,
    )


def load_experiment(
    experiment_name: str,
    experiments_path: str | Path = "experiments/configs/experiments.yaml",
    base_path: str | Path = "experiments/configs/base.yaml",
) -> ArcD4PGConfig:
    """Load an experiment configuration as ArcD4PGConfig.

    Args:
        experiment_name: Name of the experiment in experiments.yaml.
        experiments_path: Path to experiments YAML file.
        base_path: Path to base config YAML file.

    Returns:
        ArcD4PGConfig instance ready for use with ArcD4PG.

    Example:
        >>> config = load_experiment("stable_baseline")
        >>> experiment = ArcD4PG(config)
        >>> experiment.train(epochs=10)
    """
    config_dict = load_experiment_config(experiment_name, experiments_path, base_path)
    return dict_to_config(config_dict)


def list_experiments(
    experiments_path: str | Path = "experiments/configs/experiments.yaml",
) -> dict[str, str]:
    """List all available experiments with descriptions.

    Args:
        experiments_path: Path to experiments YAML file.

    Returns:
        Dictionary mapping experiment names to descriptions.
    """
    experiments = load_yaml(experiments_path)
    return {
        name: config.get("description", "No description")
        for name, config in experiments.items()
    }
