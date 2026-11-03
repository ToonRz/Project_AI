"""Synthetic competitor hotels whose hourly prices and availability our OTA customers can see."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.baselines import FixedPolicy, RuleBasedPolicy
from src.config import PROJECT_ROOT, load_config
from src.env import HotelPricingEnv
from src.hotels import Hotel, sample_synthetic_hotel


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def generate_competitor_hotels(
    config: dict[str, Any] | None = None,
    profiles: dict[str, Any] | None = None,
    mode: str | None = None,
) -> dict[str, list[Hotel]]:
    """Generate the competitor pool from its own seed so other pools stay unchanged."""
    config = config or load_config()
    if profiles is None:
        with _project_path(config["paths"]["tier_profiles"]).open("r", encoding="utf-8") as stream:
            profiles = json.load(stream)
    rng = np.random.default_rng(config["market"]["seed"])
    count = config["market"]["competitors_per_tier"]
    return {
        tier_name: [
            sample_synthetic_hotel(
                tier_name,
                f"{tier_name}_competitor_{index:02d}",
                "competitor",
                rng,
                config,
                profiles,
                mode,
            )
            for index in range(count)
        ]
        for tier_name in config["tiers"]
    }


@dataclass(frozen=True)
class CompetitorTrack:
    hotel_id: str
    strategy: str
    sizes_m2: np.ndarray  # (rooms,)
    prices: np.ndarray  # (steps, rooms) THB published at each hour
    available: np.ndarray  # (steps, rooms) bool; rooms_left > 0 after cancellations, before demand


def simulate_competitor(
    hotel: Hotel,
    strategy: str,
    season: str,
    is_weekend: bool,
    scenario_seed: int,
    config: dict[str, Any],
) -> CompetitorTrack:
    """Run one competitor in its own no-competitor world and record its hourly offers."""
    env = HotelPricingEnv(hotel=hotel, config=config)
    env.reset(
        seed=scenario_seed,
        options={
            "hotel": hotel,
            "season": season,
            "is_weekend": is_weekend,
            "scenario_seed": scenario_seed,
        },
    )
    markup_level = config["market"]["strategies"][strategy]
    policy = (
        RuleBasedPolicy(config)
        if markup_level is None
        else FixedPolicy(config, markup_level=markup_level)
    )
    prices: list[list[float]] = []
    available: list[np.ndarray] = []
    while True:
        actions = policy.select_actions(env)
        available.append(env.rooms_left > 0)
        _, _, terminated, truncated, info = env.step(actions)
        prices.append(info["prices"])
        if terminated or truncated:
            break
    return CompetitorTrack(
        hotel_id=hotel.hotel_id,
        strategy=strategy,
        sizes_m2=np.asarray([room.size_m2 for room in hotel.rooms], dtype=float),
        prices=np.asarray(prices, dtype=float),
        available=np.asarray(available, dtype=bool),
    )


class Market:
    """Hourly competitor offers, flattened once so customers can compare them cheaply."""

    def __init__(self, competitors: Sequence[CompetitorTrack]):
        self.competitors = list(competitors)
        if not self.competitors:
            self.prices_flat = np.empty((0, 0), dtype=float)
            self.sizes_flat = np.empty(0, dtype=float)
            self.available_flat = np.empty((0, 0), dtype=bool)
            self.owner_flat = np.empty(0, dtype=np.int64)
            self.median_prices = np.empty((0, 0), dtype=float)
            return
        self.prices_flat = np.concatenate([track.prices for track in self.competitors], axis=1)
        self.sizes_flat = np.concatenate([track.sizes_m2 for track in self.competitors])
        self.available_flat = np.concatenate([track.available for track in self.competitors], axis=1)
        self.owner_flat = np.concatenate(
            [np.full(len(track.sizes_m2), index, dtype=np.int64) for index, track in enumerate(self.competitors)]
        )
        offered = np.where(
            np.stack([track.available for track in self.competitors]),
            np.stack([track.prices for track in self.competitors]),
            np.nan,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # steps where no competitor is available
            self.median_prices = np.nanmedian(offered, axis=0)

    def offers(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (prices, sizes_m2, available, owner) for every competitor room at this hour."""
        if not self.competitors:
            return self.prices_flat, self.sizes_flat, self.available_flat, self.owner_flat
        return self.prices_flat[step], self.sizes_flat, self.available_flat[step], self.owner_flat

    def median_price(self, step: int, room_index: int) -> float | None:
        """Median published price of the available competitor rooms of this type, or None."""
        if not self.competitors:
            return None
        value = self.median_prices[step, room_index]
        return None if np.isnan(value) else float(value)


def build_market(
    candidates: list[Hotel],
    count_range: tuple[int, int] | list[int],
    season: str,
    is_weekend: bool,
    key: Sequence[int],
    config: dict[str, Any],
    exclude_hotel_id: str | None = None,
) -> Market:
    """Draw and simulate this episode's competitors; the same key always gives the same market."""
    market_config = config["market"]
    rng = np.random.default_rng([market_config["seed"], *key])
    eligible = [hotel for hotel in candidates if hotel.hotel_id != exclude_hotel_id]
    count = int(rng.integers(count_range[0], count_range[1] + 1))
    chosen = rng.choice(len(eligible), size=count, replace=False)
    strategy_names = list(market_config["strategies"])
    strategies = [strategy_names[int(rng.integers(0, len(strategy_names)))] for _ in range(count)]
    tracks = []
    for slot, (hotel_index, strategy) in enumerate(zip(chosen, strategies)):
        scenario_seed = int(
            np.random.SeedSequence([market_config["seed"], *key, slot]).generate_state(1)[0]
        )
        tracks.append(
            simulate_competitor(
                eligible[int(hotel_index)], strategy, season, is_weekend, scenario_seed, config
            )
        )
    return Market(tracks)
