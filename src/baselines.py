"""Fixed-price and occupancy-rule policies that respect environment masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.env import HotelPricingEnv
from src.hotels import generate_hotel_population


class FixedPolicy:
    """Keep all rooms at the configured ×1.00 level, subject to business rules."""

    name = "Fixed"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

    def select_actions(self, env: HotelPricingEnv) -> np.ndarray:
        env.prepare_step()
        actions = np.zeros(len(env.levels), dtype=np.int64)
        decided_prices: list[float] = []
        target = self.config["baselines"]["fixed_markup_level"]
        for room_index in range(len(actions)):
            lower = decided_prices[-1] if decided_prices else None
            level, price, _, _ = env.resolve_room_action(room_index, target, lower)
            actions[room_index] = level
            decided_prices.append(price)
        return actions


class RuleBasedPolicy:
    """Increase markup at configured occupancy thresholds."""

    name = "Rule-based"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

    def select_actions(self, env: HotelPricingEnv) -> np.ndarray:
        env.prepare_step()
        thresholds = self.config["baselines"]["rule_thresholds"]
        target_levels = self.config["baselines"]["rule_markup_levels"]
        actions = np.zeros(len(env.levels), dtype=np.int64)
        decided_prices: list[float] = []
        for room_index, room in enumerate(env.hotel.rooms):
            occupancy = 1.0 - env.rooms_left[room_index] / room.count
            if occupancy < thresholds[0]:
                target = target_levels[0]
            elif occupancy < thresholds[1]:
                target = target_levels[1]
            elif occupancy < thresholds[2]:
                target = target_levels[2]
            else:
                target = target_levels[3]
            lower = decided_prices[-1] if decided_prices else None
            level, price, _, _ = env.resolve_room_action(room_index, target, lower)
            actions[room_index] = level
            decided_prices.append(price)
        return actions


def default_baselines(config: dict[str, Any] | None = None) -> list[FixedPolicy | RuleBasedPolicy]:
    config = config or load_config()
    return [FixedPolicy(config), RuleBasedPolicy(config)]


def evaluate_baselines(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both baseline policies once on every unseen test hotel."""
    config = config or load_config()
    _, test_hotels = generate_hotel_population(config)
    policies = default_baselines(config)
    episodes = []

    for tier_name, hotels in test_hotels.items():
        tier_id = config["tiers"][tier_name]["id"]
        for hotel_index, hotel in enumerate(hotels):
            seed = (
                config["evaluation"]["seed_base"]
                + config["evaluation"]["seed_episode_multiplier"] * hotel_index
                + tier_id
            )
            rng = np.random.default_rng(seed)
            season = str(
                rng.choice(
                    config["episode"]["season_names"],
                    p=config["episode"]["season_probabilities"],
                )
            )
            is_weekend = bool(rng.random() < config["episode"]["weekend_probability"])
            env = HotelPricingEnv(hotel=hotel, config=config)

            for policy in policies:
                _, _ = env.reset(
                    seed=seed,
                    options={
                        "hotel": hotel,
                        "season": season,
                        "is_weekend": is_weekend,
                        "scenario_seed": seed,
                    },
                )
                while True:
                    action = policy.select_actions(env)
                    _, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        episodes.append(
                            {
                                "policy": policy.name,
                                **info["episode_metrics"],
                            }
                        )
                        break

    episode_table = pd.DataFrame(episodes)
    kpis = (
        episode_table.groupby(["policy", "tier"], as_index=False)
        .agg(
            revenue=("revenue", "mean"),
            net_profit=("net_profit", "mean"),
            occupancy=("occupancy", "mean"),
            adr=("adr", "mean"),
            revpar=("revpar", "mean"),
            profit_per_available_room=("profit_per_available_room", "mean"),
            episodes=("hotel_id", "size"),
        )
    )
    fixed_profit = kpis.loc[kpis["policy"].eq("Fixed")].set_index("tier")["net_profit"]
    kpis["net_profit_pct_vs_fixed"] = kpis.apply(
        lambda row: (row["net_profit"] / fixed_profit.loc[row["tier"]] - 1.0) * 100.0,
        axis=1,
    )
    output_dir = PROJECT_ROOT / config["paths"]["results"]
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_table.to_csv(output_dir / "baseline_episodes.csv", index=False)
    kpis.to_csv(output_dir / "baseline_kpis.csv", index=False)
    return episode_table, kpis


if __name__ == "__main__":
    _, kpis = evaluate_baselines()
    print(kpis.to_string(index=False))
