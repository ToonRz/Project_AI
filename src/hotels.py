"""Tier profiles and reproducible synthetic-hotel domain randomization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, load_config


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@dataclass
class RoomType:
    name: str
    size_m2: float
    count: int


@dataclass
class Hotel:
    hotel_id: str
    tier: str
    pool: str
    reference_price_thb: float
    business_share: float
    demand_per_room_episode: float
    variable_cost_thb: float
    ota_share_leisure: float
    ota_share_business: float
    segments: dict[str, dict[str, float]]
    rooms: list[RoomType]
    arrival_weights: dict[str, list[float]]
    season_multipliers: dict[str, dict[str, float]]
    dow_multipliers: dict[str, dict[str, float]]
    h_shift: int
    amp_season: float
    amp_weekend: float

    @property
    def capacity(self) -> int:
        return sum(room.count for room in self.rooms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profile_hotel(
    tier_name: str,
    hotel_id: str,
    pool: str,
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> Hotel:
    source = profiles["tiers"][tier_name]
    tier_config = config["tiers"][tier_name]
    return Hotel(
        hotel_id=hotel_id,
        tier=tier_name,
        pool=pool,
        reference_price_thb=float(source["reference_price_thb"]),
        business_share=float(source["business_share"]),
        demand_per_room_episode=float(source["demand_per_room_episode"]),
        variable_cost_thb=float(source["variable_cost_thb"]),
        ota_share_leisure=float(source["ota_share_leisure"]),
        ota_share_business=float(source["ota_share_business"]),
        segments={
            segment: {
                "wtp_median_reference": float(values["wtp_median_reference"]),
                "wtp_sigma": float(values["wtp_sigma"]),
                "size_beta": float(values["size_beta"]),
                "cancel_base": float(values["cancel_base"]),
            }
            for segment, values in source["segments"].items()
        },
        rooms=[
            RoomType(
                name=room["name"],
                size_m2=float(room["size_m2"]),
                count=int(room["count"]),
            )
            for room in source["rooms"]
        ],
        arrival_weights={
            segment: [float(value) for value in values]
            for segment, values in source.get("arrival_weights", {}).items()
        },
        season_multipliers=source.get(
            "season_multipliers", config["demand"]["season_multipliers"]
        ),
        dow_multipliers=source.get("dow_multipliers", config["demand"]["dow_multipliers"]),
        h_shift=0,
        amp_season=float(config["demand"]["hotel_amplitude"]["season"]),
        amp_weekend=float(config["demand"]["hotel_amplitude"]["weekend"]),
    )


def sample_synthetic_hotel(
    tier_name: str,
    hotel_id: str,
    pool: str,
    rng: np.random.Generator,
    config: dict[str, Any],
    profiles: dict[str, Any],
    mode: str | None = None,
) -> Hotel:
    """Sample one hotel using the configured multiplicative/additive spreads."""
    hotel = _profile_hotel(tier_name, hotel_id, pool, config, profiles)
    randomization = config["synthetic_hotels"]
    mode = mode or randomization["domain_randomization_mode"]
    if mode == randomization["single_profile_mode"]:
        return hotel
    if mode != randomization["domain_randomization_mode"]:
        raise ValueError(f"Unknown synthetic hotel mode: {mode}")

    spreads = profiles["tiers"][tier_name]["spreads"]
    z_reference = rng.normal()
    z_demand = rng.normal()
    z_wtp = rng.normal()
    z_sigma = rng.normal()
    z_beta = rng.normal()
    z_cancel = rng.normal()
    z_cost = rng.normal()
    z_size = rng.normal()
    z_count = rng.normal()

    hotel.reference_price_thb *= np.exp(spreads["reference_price"] * z_reference)
    hotel.demand_per_room_episode *= np.exp(spreads["demand"] * z_demand)
    hotel.variable_cost_thb *= np.exp(spreads["cost"] * z_cost)

    hotel.business_share = float(
        np.clip(
            hotel.business_share
            + randomization["additive_spreads"]["business_share"] * rng.normal(),
            randomization["additive_spreads"]["business_share_min"],
            randomization["additive_spreads"]["business_share_max"],
        )
    )
    hotel.ota_share_leisure = float(
        np.clip(
            hotel.ota_share_leisure + randomization["additive_spreads"]["ota_share"] * rng.normal(),
            0.0,
            1.0,
        )
    )
    hotel.ota_share_business = float(
        np.clip(
            hotel.ota_share_business + randomization["additive_spreads"]["ota_share"] * rng.normal(),
            0.0,
            1.0,
        )
    )

    for segment_name, segment in hotel.segments.items():
        wtp_spread = spreads["wtp"]
        if segment_name == "business":
            wtp_spread += spreads["business_wtp_extra"]
        segment["wtp_median_reference"] *= np.exp(wtp_spread * z_wtp)
        segment["wtp_sigma"] *= np.exp(spreads["sigma"] * z_sigma)
        segment["size_beta"] *= np.exp(spreads["beta"] * z_beta)
        segment["cancel_base"] = float(
            np.clip(segment["cancel_base"] * np.exp(spreads["cancel"] * z_cancel), 0.0, 1.0)
        )

    size_factor = np.exp(spreads["room_size"] * z_size)
    count_factor = np.exp(spreads["room_count"] * z_count)
    for room in hotel.rooms:
        room.size_m2 = round(room.size_m2 * size_factor, 2)
        room.count = max(
            randomization["minimum_room_count"],
            int(round(room.count * count_factor)),
        )

    amplitude = randomization["amplitude"]
    hotel.h_shift = int(round(randomization["h_shift_standard_deviation_hours"] * rng.normal()))
    hotel.amp_season = float(
        np.clip(
            amplitude["season_base"] + amplitude["season_z_scale"] * rng.normal(),
            amplitude["season_min"],
            amplitude["season_max"],
        )
    )
    hotel.amp_weekend = float(
        np.clip(
            amplitude["weekend_base"] + amplitude["weekend_z_scale"] * rng.normal(),
            amplitude["weekend_min"],
            amplitude["weekend_max"],
        )
    )
    return hotel


def generate_hotel_population(
    config: dict[str, Any] | None = None,
    profiles: dict[str, Any] | None = None,
    mode: str | None = None,
) -> tuple[dict[str, list[Hotel]], dict[str, list[Hotel]]]:
    """Generate disjoint train/test hotel pools in a fixed tier and seed order."""
    config = config or load_config()
    if profiles is None:
        with _project_path(config["paths"]["tier_profiles"]).open("r", encoding="utf-8") as stream:
            profiles = json.load(stream)
    rng = np.random.default_rng(config["synthetic_hotels"]["seed"])
    train: dict[str, list[Hotel]] = {}
    test: dict[str, list[Hotel]] = {}
    train_count = config["synthetic_hotels"]["train_per_tier"]
    test_count = config["synthetic_hotels"]["test_per_tier"]
    for tier_name in config["tiers"]:
        train[tier_name] = [
            sample_synthetic_hotel(
                tier_name,
                f"{tier_name}_train_{index:02d}",
                "train",
                rng,
                config,
                profiles,
                mode,
            )
            for index in range(train_count)
        ]
        test[tier_name] = [
            sample_synthetic_hotel(
                tier_name,
                f"{tier_name}_test_{index:02d}",
                "test",
                rng,
                config,
                profiles,
                mode,
            )
            for index in range(test_count)
        ]
    return train, test


def generate_validation_hotels(
    config: dict[str, Any] | None = None,
    profiles: dict[str, Any] | None = None,
    mode: str | None = None,
) -> dict[str, list[Hotel]]:
    """Generate a tuning pool from its own seed so train/test draws stay unchanged."""
    config = config or load_config()
    if profiles is None:
        with _project_path(config["paths"]["tier_profiles"]).open("r", encoding="utf-8") as stream:
            profiles = json.load(stream)
    rng = np.random.default_rng(config["synthetic_hotels"]["validation_seed"])
    count = config["synthetic_hotels"]["validation_per_tier"]
    return {
        tier_name: [
            sample_synthetic_hotel(
                tier_name,
                f"{tier_name}_validation_{index:02d}",
                "validation",
                rng,
                config,
                profiles,
                mode,
            )
            for index in range(count)
        ]
        for tier_name in config["tiers"]
    }


def save_hotel_population(
    train: dict[str, list[Hotel]], test: dict[str, list[Hotel]], output_path: str | Path
) -> Path:
    path = _project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    population = {
        "train": {tier: [hotel.to_dict() for hotel in hotels] for tier, hotels in train.items()},
        "test": {tier: [hotel.to_dict() for hotel in hotels] for tier, hotels in test.items()},
    }
    path.write_text(json.dumps(population, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sim_config.yaml")
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    train, test = generate_hotel_population(config=config, mode=args.mode)
    output_path = _project_path(config["paths"]["results"]) / "synthetic_hotels.json"
    save_hotel_population(train, test, output_path)
    print(f"Train hotels: {sum(map(len, train.values()))}")
    print(f"Test hotels: {sum(map(len, test.values()))}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
