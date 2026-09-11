"""Configuration loader for DigitalSpy.

Loads and validates YAML config files.  All code must read
parameters from configs — no magic numbers in source files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def system() -> dict[str, Any]:
    return _load("system")


def features() -> dict[str, Any]:
    return _load("features")


def labels() -> dict[str, Any]:
    return _load("labels")


def splits() -> dict[str, Any]:
    return _load("splits")


def markov() -> dict[str, Any]:
    return _load("markov")


def lstm() -> dict[str, Any]:
    return _load("lstm")


def resolve_path(key: str) -> Path:
    """Resolve a path key from system config relative to project root."""
    sys_cfg = system()
    raw = sys_cfg["paths"][key]
    base = _CONFIG_DIR.parent
    return (base / raw).resolve()
