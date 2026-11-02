import copy

import pandas as pd
import pytest

from src.config import load_config
from src.evaluate import _comparison_table, _stress_hotel
from src.hotels import generate_hotel_population


def test_comparison_table_uses_fixed_within_each_group():
    rows = []
    for season, fixed_profit, rl_profit in [("low", 10.0, 12.0), ("high", 20.0, 21.0)]:
        for policy, profit in [("Fixed", fixed_profit), ("RL", rl_profit)]:
            rows.append(
                {
                    "policy": policy,
                    "tier": "budget",
                    "season": season,
                    "revenue": profit,
                    "net_profit": profit,
                    "occupancy": 0.5,
                    "adr": profit,
                    "revpar": profit / 2,
                    "profit_per_available_room": profit / 2,
                    "booking_count": 1,
                    "cancellation_count": 0,
                }
            )
    result = _comparison_table(pd.DataFrame(rows), ["tier", "season"])
    rl_rows = result.loc[result["policy"].eq("RL")].set_index("season")
    assert rl_rows.loc["low", "net_profit_pct_vs_fixed"] == pytest.approx(20.0)
    assert rl_rows.loc["high", "net_profit_pct_vs_fixed"] == pytest.approx(5.0)


def test_stress_world_changes_only_requested_hotel_attributes():
    config = load_config()
    train, _ = generate_hotel_population(config)
    source = train["budget"][0]
    original = copy.deepcopy(source)
    stressed = _stress_hotel(
        source,
        {"wtp": 0.85, "demand": 0.75, "sigma": 0.7},
        "stress",
    )
    assert stressed.demand_per_room_episode == original.demand_per_room_episode * 0.75
    assert stressed.segments["leisure"]["wtp_median_reference"] == (
        original.segments["leisure"]["wtp_median_reference"] * 0.85
    )
    assert stressed.segments["leisure"]["wtp_sigma"] == original.segments["leisure"]["wtp_sigma"] * 0.7
    assert source.demand_per_room_episode == original.demand_per_room_episode
