"""Common checkpointing utilities.

Provides schema versioning and path resolution helpers for modules that
support saving to either a file path or a directory path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCHEMA_VERSION: int = 1


def resolve_save_path(path: str | Path, class_name: str) -> Path:
    """Return a concrete file path for saving a checkpoint.

    If ``path`` is a directory or has no suffix, create the directory and
    return ``<path>/<class_name>.ptc``. Otherwise, return the file path and
    ensure its parent directory exists.

    Args:
        path: Path to save the checkpoint to.
        class_name: Name of the class to save.

    Returns:
        Path to the checkpoint file.
    """
    p = Path(path)
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{class_name}.ptc"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_load_path(path: str | Path, class_name: str) -> Path:
    """Return a concrete file path for loading a checkpoint.

    If ``path`` is a directory or has no suffix, resolve to
    ``<path>/<class_name>.ptc``. Otherwise, return the file path.

    Args:
        path: Path to load the checkpoint from.
        class_name: Name of the class to load.

    Returns:
        Path to the checkpoint file.
    """
    p = Path(path)
    if p.is_dir() or p.suffix == "":
        return p / f"{class_name}.ptc"
    return p


def normalize_class_meta(value: Any) -> str | None:
    """Return a normalized class name from mixed legacy representations.

    Accepts either a class object, a class name string, or None and returns
    the class name string (or None if not available).
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return value.__name__
    except Exception:
        return None


def assert_expected_class(meta_value: Any, expected_class_name: str) -> None:
    """Validate that checkpoint meta identifies the expected class.

    Accepts either class object or string and compares by class name. If
    the meta is missing or does not match, raises ValueError.

    Args:
        meta_value: Class object or string.
        expected_class_name: Expected class name.
    """
    cls_name = normalize_class_meta(meta_value)
    if cls_name is None:
        raise ValueError("Checkpoint is missing class metadata.")
    if cls_name != expected_class_name:
        raise ValueError(
            f"Checkpoint mismatch: expected {expected_class_name}, got {cls_name}."
        )


def add_schema_and_class(payload: dict[str, Any], class_name: str) -> dict[str, Any]:
    """Attach schema_version and class name to a payload dict.

    Returns a shallow copy with the two keys added.

    Args:
        payload: Payload dict to attach schema_version and class name to.
        class_name: Name of the class to attach.

    Returns:
        Payload dict with schema_version and class name attached.
    """
    out = dict(payload)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("class", class_name)
    return out
