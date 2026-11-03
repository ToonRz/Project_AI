"""Evaluate pricing policies on common, disjoint synthetic-hotel scenarios."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines import FixedPolicy, RuleBasedPolicy
from src.config import PROJECT_ROOT, load_config
from src.env import HotelPricingEnv
from src.hotels import Hotel, generate_hotel_population, generate_validation_hotels
from src.market import Market, build_market, generate_competitor_hotels
from src.rl_agent import QTablePolicy, TabularQLearner


METRICS = [
    "revenue",
    "net_profit",
    "occupancy",
    "adr",
    "revpar",
    "profit_per_available_room",
    "booking_count",
    "cancellation_count",
]


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_update_counts(model_path: str | Path) -> np.ndarray | None:
    path = _project_path(model_path)
    counts_name = path.name.replace("q_table", "q_counts", 1)
    counts_path = path.with_name(counts_name)
    return np.load(counts_path) if counts_path.exists() else None


def _comparison_table(episodes: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    keys = ["policy", *group_columns]
    grouped = episodes.groupby(keys, as_index=False, dropna=False)[METRICS].mean()
    counts = episodes.groupby(keys, dropna=False).size().rename("episodes").reset_index()
    grouped = grouped.merge(counts, on=keys, how="left")
    reference_keys = group_columns
    fixed = grouped.loc[grouped["policy"].eq("Fixed")]
    if reference_keys:
        fixed_profit = fixed.set_index(reference_keys)["net_profit"]
        grouped["net_profit_pct_vs_fixed"] = grouped.apply(
            lambda row: (
                row["net_profit"] / fixed_profit.loc[tuple(row[key] for key in reference_keys)] - 1.0
            ) * 100.0,
            axis=1,
        )
    else:
        fixed_profit = float(fixed["net_profit"].iloc[0])
        grouped["net_profit_pct_vs_fixed"] = (grouped["net_profit"] / fixed_profit - 1.0) * 100.0
    return grouped


def goal_table(episodes: pd.DataFrame, group_columns: list[str], margin: float) -> pd.DataFrame:
    """Profit, occupancy-floor and customer-outcome summary per policy and group."""
    keys = ["policy", *group_columns]
    grouped = episodes.groupby(keys, as_index=False, dropna=False).agg(
        net_profit=("net_profit", "mean"),
        occupancy=("occupancy", "mean"),
        booked=("booked", "sum"),
        too_expensive=("too_expensive", "sum"),
        full=("full", "sum"),
        competitor=("competitor", "sum"),
        ota_booked=("ota_booked", "sum"),
        episodes=("net_profit", "size"),
    )
    grouped["conversion"] = grouped["booked"] / (
        grouped["booked"] + grouped["too_expensive"] + grouped["competitor"]
    )
    grouped["ota_market_share"] = grouped["ota_booked"] / (grouped["ota_booked"] + grouped["competitor"])
    fixed = (
        grouped.loc[grouped["policy"].eq("Fixed"), [*group_columns, "net_profit", "occupancy"]]
        .rename(columns={"net_profit": "fixed_net_profit", "occupancy": "fixed_occupancy"})
    )
    grouped = grouped.merge(fixed, on=group_columns, how="left")
    grouped["net_profit_pct_vs_fixed"] = (grouped["net_profit"] / grouped["fixed_net_profit"] - 1.0) * 100.0
    grouped["occupancy_floor"] = grouped["fixed_occupancy"] - margin
    grouped["meets_floor"] = grouped["occupancy"] >= grouped["occupancy_floor"] - 1e-12
    return grouped.drop(columns=["fixed_net_profit", "fixed_occupancy"])


def _make_policies(
    q_table: np.ndarray,
    config: dict[str, Any],
    seed: int,
    update_counts: np.ndarray | None = None,
) -> list[Any]:
    learner = TabularQLearner(
        config,
        algorithm="n_step",
        q_table=q_table,
        update_counts=update_counts,
    )
    return [
        FixedPolicy(config),
        RuleBasedPolicy(config),
        QTablePolicy(learner, config, seed=seed),
    ]


def _episode_context(seed: int, config: dict[str, Any]) -> tuple[str, bool]:
    rng = np.random.default_rng(seed)
    season = str(
        rng.choice(
            config["episode"]["season_names"],
            p=config["episode"]["season_probabilities"],
        )
    )
    weekend = bool(rng.random() < config["episode"]["weekend_probability"])
    return season, weekend


def _run_one_scenario(
    hotel: Hotel,
    seed: int,
    season: str,
    weekend: bool,
    policies: list[Any],
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    capture_replay: bool = False,
    market: Market | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    customer_rows: list[dict[str, Any]] = []
    metadata = metadata or {}

    for policy in policies:
        env = HotelPricingEnv(hotel=hotel, config=config)
        env.reset(
            seed=seed,
            options={
                "hotel": hotel,
                "season": season,
                "is_weekend": weekend,
                "scenario_seed": seed,
                "market": market,
            },
        )
        outcomes = {"booked": 0, "too_expensive": 0, "full": 0, "competitor": 0, "ota_booked": 0}
        daily_bookings: dict[tuple[int, int], int] = {}
        step_index = 0
        while True:
            action = policy.select_actions(env)
            _, _, terminated, truncated, info = env.step(action)
            for customer in info["customer_log"]:
                outcomes[customer["status"]] += 1
                outcomes["ota_booked"] += customer["status"] == "booked" and customer["via_ota"]
            if capture_replay:
                elapsed_day = step_index // config["episode"]["hours_per_day"] + 1
                hour = step_index % config["episode"]["hours_per_day"]
                for room_index, room in enumerate(hotel.rooms):
                    hourly_rows.append(
                        {
                            "policy": policy.name,
                            "tier": hotel.tier,
                            "hotel_id": hotel.hotel_id,
                            "scenario_seed": seed,
                            "step": step_index,
                            "sale_day": elapsed_day,
                            "days_remaining": info["days_remaining"],
                            "hour_of_day": hour,
                            "room_index": room_index,
                            "room_name": room.name,
                            "price_thb": info["prices"][room_index],
                            "action_level": info["action_levels"][room_index],
                            "rooms_left": info["rooms_left"][room_index],
                            "new_bookings": sum(
                                booking["room_index"] == room_index for booking in info["bookings"]
                            ),
                            "cancellations": sum(
                                cancellation["room_index"] == room_index
                                for cancellation in info["cancellations"]
                            ),
                            "reward": info["per_room_rewards"][room_index],
                        }
                    )
                for booking in info["bookings"]:
                    key = (elapsed_day, booking["room_index"])
                    daily_bookings[key] = daily_bookings.get(key, 0) + 1
                for customer in info["customer_log"]:
                    customer_rows.append(
                        {
                            "policy": policy.name,
                            "tier": hotel.tier,
                            "hotel_id": hotel.hotel_id,
                            "scenario_seed": seed,
                            "step": step_index,
                            "sale_day": elapsed_day,
                            "hour_of_day": hour,
                            **customer,
                        }
                    )
            step_index += 1
            if terminated or truncated:
                episode_rows.append(
                    {
                        "policy": policy.name,
                        "scenario_seed": seed,
                        "season": season,
                        "is_weekend": weekend,
                        **metadata,
                        **info["episode_metrics"],
                        **outcomes,
                    }
                )
                if capture_replay:
                    for (sale_day, room_index), bookings in daily_bookings.items():
                        daily_rows.append(
                            {
                                "policy": policy.name,
                                "tier": hotel.tier,
                                "hotel_id": hotel.hotel_id,
                                "scenario_seed": seed,
                                "sale_day": sale_day,
                                "room_index": room_index,
                                "room_name": hotel.rooms[room_index].name,
                                "bookings": bookings,
                            }
                        )
                break
    return episode_rows, hourly_rows, daily_rows, customer_rows


def _evaluate_pool(
    q_table: np.ndarray,
    test_hotels: dict[str, list[Hotel]],
    config: dict[str, Any],
    episodes_per_tier: int,
    seed_base: int,
    seed_multiplier: int,
    model_label: str,
    update_counts: np.ndarray | None = None,
    competitor_hotels: dict[str, list[Hotel]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policies = _make_policies(
        q_table,
        config,
        seed=config["evaluation"]["seed_base"],
        update_counts=update_counts,
    )
    episode_rows: list[dict[str, Any]] = []
    market_enabled = config["market"]["enabled"]
    if market_enabled:
        competitors = competitor_hotels or generate_competitor_hotels(config)
        competitor_count = config["market"]["eval_competitor_count"]

    for tier_name, hotels in test_hotels.items():
        tier_id = config["tiers"][tier_name]["id"]
        for episode_index in range(episodes_per_tier):
            hotel = hotels[episode_index % len(hotels)]
            seed = seed_base + seed_multiplier * episode_index + tier_id
            season, weekend = _episode_context(seed, config)
            market = (
                build_market(
                    competitors[tier_name],
                    (competitor_count, competitor_count),
                    season,
                    weekend,
                    (seed,),
                    config,
                )
                if market_enabled
                else None
            )
            rows, _, _, _ = _run_one_scenario(
                hotel,
                seed,
                season,
                weekend,
                policies,
                config,
                metadata={"model": model_label, "episode_index": episode_index},
                market=market,
            )
            episode_rows.extend(rows)

    episodes = pd.DataFrame(episode_rows)
    overall = _comparison_table(episodes, ["model", "tier"])
    by_season = _comparison_table(episodes, ["model", "tier", "season"])
    by_weekend = _comparison_table(episodes, ["model", "tier", "is_weekend"])
    return episodes, overall, by_season, by_weekend


def evaluate_agents(
    q_table: np.ndarray,
    config: dict[str, Any] | None = None,
    episodes_per_tier: int | None = None,
    output_stem: str = "evaluation",
    model_label: str = "n_step_domain_randomization",
    update_counts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Fixed, Rule-based, and greedy tabular policies on common scenarios."""
    config = config or load_config()
    episodes_per_tier = episodes_per_tier or config["evaluation"]["episodes_per_tier"]
    if config["evaluation"]["test_hotel_pool"] == "validation":
        hotels = generate_validation_hotels(config)
        seed_base = config["evaluation"]["validation_seed_base"]
    else:
        _, hotels = generate_hotel_population(config)
        seed_base = config["evaluation"]["seed_base"]
    episodes, kpis, by_season, by_weekend = _evaluate_pool(
        q_table,
        hotels,
        config,
        episodes_per_tier,
        seed_base,
        config["evaluation"]["seed_episode_multiplier"],
        model_label,
        update_counts=update_counts,
    )
    output_dir = _project_path(config["paths"]["results"])
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output_dir / f"{output_stem}_episodes.csv", index=False)
    kpis.to_csv(output_dir / f"{output_stem}_kpis.csv", index=False)
    goal_table(episodes, ["model", "tier"], config["rl"]["occupancy_floor_margin"]).to_csv(
        output_dir / f"{output_stem}_goals.csv", index=False
    )
    by_season.to_csv(output_dir / f"{output_stem}_by_season.csv", index=False)
    by_weekend.to_csv(output_dir / f"{output_stem}_by_weekend.csv", index=False)
    return episodes, kpis


def evaluate_model_suite(
    model_paths: dict[str, str | Path], config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare available n-step and Monte Carlo single/DR models at full scale."""
    config = config or load_config()
    _, test_hotels = generate_hotel_population(config)
    episode_tables: list[pd.DataFrame] = []
    output_dir = _project_path(config["paths"]["results"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, path in model_paths.items():
        model_path = _project_path(path)
        q_table = np.load(model_path)
        update_counts = _load_update_counts(model_path)
        episodes, overall, by_season, by_weekend = _evaluate_pool(
            q_table,
            test_hotels,
            config,
            config["evaluation"]["episodes_per_tier"],
            config["evaluation"]["seed_base"],
            config["evaluation"]["seed_episode_multiplier"],
            label,
            update_counts=update_counts,
        )
        episodes.to_csv(output_dir / f"phase6_{label}_episodes.csv", index=False)
        overall.to_csv(output_dir / f"phase6_{label}_kpis.csv", index=False)
        by_season.to_csv(output_dir / f"phase6_{label}_by_season.csv", index=False)
        by_weekend.to_csv(output_dir / f"phase6_{label}_by_weekend.csv", index=False)
        generate_policy_heatmap(
            q_table,
            config,
            model_label=label,
            update_counts=update_counts,
        )
        episode_tables.append(episodes)
        print(f"Completed full evaluation for {label}", flush=True)

    combined = pd.concat(episode_tables, ignore_index=True)
    all_kpis = _comparison_table(combined, ["model", "tier"])
    all_kpis.to_csv(output_dir / "phase6_main_kpis.csv", index=False)
    combined.to_csv(output_dir / "phase6_main_episodes.csv", index=False)
    return combined, all_kpis


def _stress_hotel(hotel: Hotel, world: dict[str, float], suffix: str) -> Hotel:
    stressed = copy.deepcopy(hotel)
    stressed.hotel_id = f"{hotel.hotel_id}_{suffix}"
    stressed.demand_per_room_episode *= world["demand"]
    for segment in stressed.segments.values():
        segment["wtp_median_reference"] *= world["wtp"]
        segment["wtp_sigma"] *= world["sigma"]
    return stressed


def run_stress_tests(
    q_table: np.ndarray,
    config: dict[str, Any] | None = None,
    update_counts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare policies under the five configured demand/valuation stress worlds."""
    config = config or load_config()
    _, test_hotels = generate_hotel_population(config)
    policies = _make_policies(
        q_table, config, config["evaluation"]["seed_base"], update_counts
    )
    rows: list[dict[str, Any]] = []
    hotel_count = config["evaluation"]["stress_hotels_per_tier"]
    seed_base = config["evaluation"]["stress_seed_base"]
    market_enabled = config["market"]["enabled"]
    if market_enabled:
        competitor_pool = generate_competitor_hotels(config)
        competitor_count = config["market"]["eval_competitor_count"]
    for world_name, world in config["evaluation"]["stress_worlds"].items():
        for tier_name, hotels in test_hotels.items():
            tier_id = config["tiers"][tier_name]["id"]
            if market_enabled:
                stressed_competitors = [
                    _stress_hotel(competitor, world, world_name)
                    for competitor in competitor_pool[tier_name]
                ]
            for hotel_index in range(hotel_count):
                hotel = _stress_hotel(hotels[hotel_index % len(hotels)], world, world_name)
                seed = seed_base + tier_id
                season, weekend = _episode_context(seed, config)
                market = (
                    build_market(
                        stressed_competitors,
                        (competitor_count, competitor_count),
                        season,
                        weekend,
                        (seed, hotel_index),
                        config,
                    )
                    if market_enabled
                    else None
                )
                episode_rows, _, _, _ = _run_one_scenario(
                    hotel,
                    seed,
                    season,
                    weekend,
                    policies,
                    config,
                    metadata={"world": world_name, "hotel_index": hotel_index},
                    market=market,
                )
                rows.extend(episode_rows)
    episodes = pd.DataFrame(rows)
    kpis = _comparison_table(episodes, ["world", "tier"])
    output_dir = _project_path(config["paths"]["results"])
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output_dir / "phase6_stress_episodes.csv", index=False)
    kpis.to_csv(output_dir / "phase6_stress_kpis.csv", index=False)
    goal_table(episodes, ["world", "tier"], config["rl"]["occupancy_floor_margin"]).to_csv(
        output_dir / "phase6_stress_goals.csv", index=False
    )
    return episodes, kpis


def _load_profiles(config: dict[str, Any]) -> dict[str, Any]:
    path = _project_path(config["paths"]["tier_profiles"])
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _scaled_spread_inputs(
    config: dict[str, Any], factor: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    scaled_config = copy.deepcopy(config)
    randomization = scaled_config["synthetic_hotels"]
    for key in randomization["multiplicative_spreads"]:
        randomization["multiplicative_spreads"][key] *= factor
    for key in ["business_share", "ota_share"]:
        randomization["additive_spreads"][key] *= factor
    randomization["h_shift_standard_deviation_hours"] *= factor
    randomization["amplitude"]["season_z_scale"] *= factor
    randomization["amplitude"]["weekend_z_scale"] *= factor

    profiles = copy.deepcopy(_load_profiles(config))
    for tier in profiles["tiers"].values():
        for key in tier["spreads"]:
            tier["spreads"][key] *= factor
    return scaled_config, profiles


def _scaled_demand_hotel(hotel: Hotel, base_conversion: float, target_conversion: float) -> Hotel:
    scaled = copy.deepcopy(hotel)
    scaled.hotel_id = f"{hotel.hotel_id}_conversion_{target_conversion:.2f}"
    scaled.demand_per_room_episode *= base_conversion / target_conversion
    return scaled


def run_sensitivity(
    q_table: np.ndarray,
    config: dict[str, Any] | None = None,
    update_counts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one-factor conversion-rate and hotel-spread sensitivity experiments."""
    config = config or load_config()
    conversion_values = config["data"]["conversion_rate_sensitivity"]
    spread_values = config["evaluation"]["spread_sensitivity_factors"]
    base_conversion = config["data"]["conversion_rate"]
    policies = _make_policies(
        q_table, config, config["evaluation"]["seed_base"], update_counts
    )
    cases: list[tuple[str, float, dict[str, list[Hotel]], dict[str, list[Hotel]] | None]] = []
    market_enabled = config["market"]["enabled"]
    competitor_count = config["market"]["eval_competitor_count"]

    _, base_test_hotels = generate_hotel_population(config)
    base_competitors = generate_competitor_hotels(config) if market_enabled else None
    for conversion in conversion_values:
        hotels = {
            tier: [
                _scaled_demand_hotel(hotel, base_conversion, conversion)
                for hotel in tier_hotels
            ]
            for tier, tier_hotels in base_test_hotels.items()
        }
        competitors = (
            {
                tier: [
                    _scaled_demand_hotel(hotel, base_conversion, conversion)
                    for hotel in tier_hotels
                ]
                for tier, tier_hotels in base_competitors.items()
            }
            if market_enabled
            else None
        )
        cases.append(("conversion_rate", float(conversion), hotels, competitors))
    for factor in spread_values:
        spread_config, spread_profiles = _scaled_spread_inputs(config, float(factor))
        _, test_hotels = generate_hotel_population(
            spread_config,
            profiles=spread_profiles,
        )
        competitors = (
            generate_competitor_hotels(spread_config, profiles=spread_profiles)
            if market_enabled
            else None
        )
        cases.append(("spread_factor", float(factor), test_hotels, competitors))

    rows: list[dict[str, Any]] = []
    hotel_count = config["evaluation"]["stress_hotels_per_tier"]
    for parameter, value, test_hotels, case_competitors in cases:
        for tier_name, hotels in test_hotels.items():
            tier_id = config["tiers"][tier_name]["id"]
            for hotel_index in range(hotel_count):
                hotel = hotels[hotel_index % len(hotels)]
                seed = (
                    config["evaluation"]["seed_base"]
                    + config["evaluation"]["seed_episode_multiplier"] * hotel_index
                    + tier_id
                )
                season, weekend = _episode_context(seed, config)
                market = (
                    build_market(
                        case_competitors[tier_name],
                        (competitor_count, competitor_count),
                        season,
                        weekend,
                        (seed, hotel_index),
                        config,
                    )
                    if market_enabled
                    else None
                )
                episode_rows, _, _, _ = _run_one_scenario(
                    hotel,
                    seed,
                    season,
                    weekend,
                    policies,
                    config,
                    metadata={"parameter": parameter, "value": value, "hotel_index": hotel_index},
                    market=market,
                )
                rows.extend(episode_rows)
    episodes = pd.DataFrame(rows)
    kpis = _comparison_table(episodes, ["parameter", "value", "tier"])
    output_dir = _project_path(config["paths"]["results"])
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output_dir / "phase6_sensitivity_episodes.csv", index=False)
    kpis.to_csv(output_dir / "phase6_sensitivity_kpis.csv", index=False)
    goal_table(episodes, ["parameter", "value", "tier"], config["rl"]["occupancy_floor_margin"]).to_csv(
        output_dir / "phase6_sensitivity_goals.csv", index=False
    )
    return episodes, kpis


def _representative_day(bucket: list[int]) -> int:
    return (bucket[0] + bucket[1]) // 2


def generate_policy_heatmap(
    q_table: np.ndarray,
    config: dict[str, Any] | None = None,
    model_label: str = "n_step_domain_randomization",
    update_counts: np.ndarray | None = None,
) -> pd.DataFrame:
    """Query greedy prices for representative tabular states and save long-form data."""
    config = config or load_config()
    train_hotels, _ = generate_hotel_population(
        config, mode=config["synthetic_hotels"]["single_profile_mode"]
    )
    learner = TabularQLearner(
        config,
        algorithm="n_step",
        q_table=q_table,
        update_counts=update_counts,
    )
    policy = QTablePolicy(learner, config, seed=config["project"]["random_seed"])
    states: list[dict[str, Any]] = []
    for tier_name, hotel_list in train_hotels.items():
        hotel = hotel_list[0]
        env = HotelPricingEnv(hotel=hotel, config=config)
        for season in config["episode"]["season_names"]:
            for weekend in [False, True]:
                for day_bucket_index, day_range in enumerate(config["rl"]["state"]["day_bucket_ranges"]):
                    days_remaining = _representative_day(day_range)
                    for period in range(config["rl"]["state"]["clock_periods_per_day"]):
                        env.reset(
                            seed=config["project"]["random_seed"],
                            options={"hotel": hotel, "season": season, "is_weekend": weekend},
                        )
                        elapsed_days = config["episode"]["booking_horizon_days"] - days_remaining
                        env.step_index = elapsed_days * config["episode"]["hours_per_day"] + (
                            period * config["rl"]["state"]["time_period_hours"]
                        )
                        env.hours_remaining = config["episode"]["steps"] - env.step_index
                        remaining_buckets = config["rl"]["state"]["remaining_bucket_count"]
                        representative_bucket = config["evaluation"]["heatmap_remaining_bucket"]
                        env.rooms_left = np.asarray(
                            [
                                int(round(room.count * representative_bucket / remaining_buckets))
                                for room in hotel.rooms
                            ],
                            dtype=np.int64,
                        )
                        env.levels = np.full(
                            len(hotel.rooms),
                            config["pricing"]["initial_price_level"],
                            dtype=np.int64,
                        )
                        env.prices = np.asarray(
                            [env.price_for_level(index, int(env.levels[index])) for index in range(len(env.levels))],
                            dtype=float,
                        )
                        env.is_weekend = weekend
                        actions = policy.select_actions(env)
                        prices: list[float] = []
                        for room_index, action in enumerate(actions):
                            lower_price = prices[-1] if prices else None
                            _, price, _, _ = env.resolve_room_action(
                                room_index, int(action), lower_price
                            )
                            prices.append(price)
                            states.append(
                                {
                                    "tier": tier_name,
                                    "season": season,
                                    "is_weekend": weekend,
                                    "day_bucket": day_bucket_index,
                                    "days_remaining_representative": days_remaining,
                                    "period": period,
                                    "room_index": room_index,
                                    "room_name": hotel.rooms[room_index].name,
                                    "level": int(action),
                                    "price_thb": price,
                                    "reference_room_price_thb": env.room_reference_prices[room_index],
                                    "markup": price / env.room_reference_prices[room_index],
                                }
                            )
    heatmap = pd.DataFrame(states)
    output_dir = _project_path(config["paths"]["results"])
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap.to_csv(output_dir / f"phase6_{model_label}_policy_heatmap.csv", index=False)
    if model_label == "n_step_domain_randomization":
        heatmap.to_csv(output_dir / "phase6_policy_heatmap.csv", index=False)
    return heatmap


def generate_replay(
    q_table: np.ndarray,
    config: dict[str, Any] | None = None,
    update_counts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save one same-scenario hourly, daily, and customer-log replay per tier."""
    config = config or load_config()
    _, test_hotels = generate_hotel_population(config)
    policies = _make_policies(
        q_table, config, config["evaluation"]["seed_base"], update_counts
    )
    hourly: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    customers: list[dict[str, Any]] = []
    market_enabled = config["market"]["enabled"]
    competitor_rows: list[dict[str, Any]] = []
    if market_enabled:
        competitor_pool = generate_competitor_hotels(config)
        competitor_count = config["market"]["eval_competitor_count"]
    for tier_name, hotels in test_hotels.items():
        tier_id = config["tiers"][tier_name]["id"]
        hotel = hotels[config["evaluation"]["policy_replay_hotel_index"] % len(hotels)]
        seed = (
            config["evaluation"]["seed_base"]
            + config["evaluation"]["seed_episode_multiplier"]
            * config["evaluation"]["policy_replay_episode"]
            + tier_id
        )
        season, weekend = _episode_context(seed, config)
        market = None
        if market_enabled:
            market = build_market(
                competitor_pool[tier_name],
                (competitor_count, competitor_count),
                season,
                weekend,
                (seed,),
                config,
            )
            for competitor_index, track in enumerate(market.competitors):
                for step, room_index in np.ndindex(track.prices.shape):
                    competitor_rows.append(
                        {
                            "tier": tier_name,
                            "competitor_index": competitor_index,
                            "hotel_id": track.hotel_id,
                            "strategy": track.strategy,
                            "step": step,
                            "room_index": room_index,
                            "price_thb": float(track.prices[step, room_index]),
                            "available": bool(track.available[step, room_index]),
                            "size_m2": float(track.sizes_m2[room_index]),
                        }
                    )
        _, hour_rows, day_rows, customer_rows = _run_one_scenario(
            hotel,
            seed,
            season,
            weekend,
            policies,
            config,
            capture_replay=True,
            market=market,
        )
        hourly.extend(hour_rows)
        daily.extend(day_rows)
        customers.extend(customer_rows)
    output_dir = _project_path(config["paths"]["results"])
    hourly_table = pd.DataFrame(hourly)
    daily_table = pd.DataFrame(daily)
    customer_table = pd.DataFrame(customers)
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_table.to_csv(output_dir / "phase6_replay_hourly.csv", index=False)
    daily_table.to_csv(output_dir / "phase6_replay_daily.csv", index=False)
    customer_table.to_csv(output_dir / "phase6_replay_customer_log.csv", index=False)
    if market_enabled:
        pd.DataFrame(competitor_rows).to_csv(output_dir / "phase6_replay_competitors.csv", index=False)
    return hourly_table, daily_table, customer_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-table", default="models/q_table.npy")
    parser.add_argument("--episodes-per-tier", type=int, default=None)
    parser.add_argument("--output-stem", default="evaluation")
    parser.add_argument("--model-label", default="n_step_domain_randomization")
    parser.add_argument("--models", action="store_true", help="Run full configured model suite")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--pool", choices=["test", "validation"], default=None)
    parser.add_argument("--config", default="config/sim_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pool:
        config["evaluation"]["test_hotel_pool"] = args.pool
    q_path = _project_path(args.q_table)
    q_table = np.load(q_path)
    update_counts = _load_update_counts(q_path)
    if args.models:
        model_paths = config["evaluation"]["model_variants"]
        _, kpis = evaluate_model_suite(model_paths, config)
        print(kpis.to_string(index=False))
    else:
        _, kpis = evaluate_agents(
            q_table,
            config,
            episodes_per_tier=args.episodes_per_tier,
            output_stem=args.output_stem,
            model_label=args.model_label,
            update_counts=update_counts,
        )
        print(kpis.to_string(index=False))
    if args.stress:
        _, kpis = run_stress_tests(q_table, config, update_counts)
        print("Stress tests")
        print(kpis.to_string(index=False))
    if args.sensitivity:
        _, kpis = run_sensitivity(q_table, config, update_counts)
        print("Sensitivity")
        print(kpis.to_string(index=False))
    if args.heatmap:
        print(
            "Heatmap rows: "
            f"{len(generate_policy_heatmap(q_table, config, update_counts=update_counts))}"
        )
    if args.replay:
        hourly, daily, customer_log = generate_replay(q_table, config, update_counts)
        print(f"Replay rows: hourly={len(hourly)}, daily={len(daily)}, customers={len(customer_log)}")


if __name__ == "__main__":
    main()
