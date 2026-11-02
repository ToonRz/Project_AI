"""Shared configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sim_config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the single project configuration file."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early when core simulation dimensions are inconsistent."""
    episode = config["episode"]
    steps = episode["booking_horizon_days"] * episode["hours_per_day"]
    if episode["steps"] != steps:
        raise ValueError(f"episode.steps must equal days × hours_per_day ({steps})")
    if len(config["tiers"]) != 3:
        raise ValueError("Exactly three hotel tiers are required")
    for tier_name, tier in config["tiers"].items():
        if len(tier["rooms"]) != 3:
            raise ValueError(f"Tier {tier_name} must define exactly three room types")
        if sorted(room["size_m2"] for room in tier["rooms"]) != [
            room["size_m2"] for room in tier["rooms"]
        ]:
            raise ValueError(f"Rooms in tier {tier_name} must be ordered by size")
