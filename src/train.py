"""Train the configured tabular pricing policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines import FixedPolicy
from src.config import PROJECT_ROOT, load_config
from src.env import HotelPricingEnv
from src.hotels import generate_hotel_population
from src.market import build_market
from src.rl_agent import TabularQLearner


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def occupancy_penalties(
    rooms_left: np.ndarray, capacity: int, target_occupancy: float, penalty_per_room_thb: float
) -> np.ndarray:
    """Terminal penalty for rooms sold below target, split by each room type's unsold rooms."""
    rooms_left = np.asarray(rooms_left, dtype=float)
    shortfall = max(0.0, target_occupancy * capacity - (capacity - rooms_left.sum()))
    if shortfall == 0.0 or rooms_left.sum() == 0.0:
        return np.zeros_like(rooms_left)
    return penalty_per_room_thb * shortfall * rooms_left / rooms_left.sum()


def _fixed_occupancy(hotel: Any, options: dict[str, Any], config: dict[str, Any]) -> float:
    """Occupancy the Fixed baseline reaches on the same scenario."""
    env = HotelPricingEnv(hotel=hotel, config=config)
    env.reset(seed=options["scenario_seed"], options=options)
    policy = FixedPolicy(config)
    while True:
        _, _, terminated, truncated, info = env.step(policy.select_actions(env))
        if terminated or truncated:
            return float(info["episode_metrics"]["occupancy"])


def train_agent(
    config: dict[str, Any] | None = None,
    algorithm: str = "n_step",
    mode: str | None = None,
    episodes_per_tier: int | None = None,
    learner: TabularQLearner | None = None,
    tiers: list[str] | None = None,
    behavior_policy: Any | None = None,
    epsilon_start: float = 1.0,
) -> tuple[TabularQLearner, pd.DataFrame, Path]:
    """Train one shared table, alternating tiers and sampling configured scenarios.

    Optional warm-start: pass an existing ``learner`` to continue its table, ``tiers`` to train
    only those tier slices, and ``behavior_policy`` to take that policy's actions while the
    learner updates from them (learning from demonstration).
    """
    config = config or load_config()
    mode = mode or config["synthetic_hotels"]["domain_randomization_mode"]
    if mode not in {
        config["synthetic_hotels"]["single_profile_mode"],
        config["synthetic_hotels"]["domain_randomization_mode"],
    }:
        raise ValueError(f"Unsupported hotel sampling mode: {mode}")
    episodes_per_tier = episodes_per_tier or config["rl"]["training_episodes_per_tier"]
    tier_names = tiers or list(config["tiers"])
    total_episodes = episodes_per_tier * len(tier_names)
    seed = config["project"]["random_seed"]
    scenario_rng = np.random.default_rng(seed)
    learner = learner or TabularQLearner(config, algorithm=algorithm)
    train_hotels, _ = generate_hotel_population(config, mode=mode)
    hotel_index_rng = np.random.default_rng(config["synthetic_hotels"]["seed"])
    base_hotel = train_hotels[next(iter(config["tiers"]))][0]
    env = HotelPricingEnv(hotel=base_hotel, config=config)
    episode_records: list[dict[str, Any]] = []
    model_dir = _project_path(config["paths"]["models"])
    results_dir = _project_path(config["paths"]["results"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in range(total_episodes):
        tier_name = tier_names[episode_index % len(tier_names)]
        if mode == config["synthetic_hotels"]["single_profile_mode"]:
            hotel = train_hotels[tier_name][0]
        else:
            hotel = train_hotels[tier_name][
                int(hotel_index_rng.integers(0, len(train_hotels[tier_name])))
            ]
        season = str(
            scenario_rng.choice(
                config["episode"]["season_names"],
                p=config["episode"]["season_probabilities"],
            )
        )
        is_weekend = bool(
            scenario_rng.random() < config["episode"]["weekend_probability"]
        )
        scenario_seed = seed + episode_index
        options = {
            "hotel": hotel,
            "season": season,
            "is_weekend": is_weekend,
            "scenario_seed": scenario_seed,
        }
        if config["market"]["enabled"]:
            options["market"] = build_market(
                train_hotels[tier_name],
                config["market"]["train_competitor_count"],
                season,
                is_weekend,
                (scenario_seed,),
                config,
                exclude_hotel_id=hotel.hotel_id,
            )
        env.reset(seed=scenario_seed, options=options)
        penalty_rate = config["rl"]["occupancy_penalty_per_room"] * hotel.reference_price_thb
        target_occupancy = (
            _fixed_occupancy(hotel, options, config) - config["rl"]["occupancy_floor_margin"]
            if penalty_rate > 0
            else 0.0
        )
        occupancy_penalty = 0.0
        learner.begin_episode()
        epsilon = learner.epsilon(episode_index, total_episodes, epsilon_start)
        episode_return = 0.0

        while True:
            cancellation_rewards, _ = env.prepare_step()
            for room_index, cancellation_reward in enumerate(cancellation_rewards):
                learner.add_reward_to_last_transition(room_index, float(cancellation_reward))
            behavior_actions = behavior_policy.select_actions(env) if behavior_policy else None
            actions = np.zeros(len(env.levels), dtype=np.int64)
            states = []
            decided_prices: list[float] = []
            for room_index in range(len(actions)):
                lower_price = decided_prices[-1] if decided_prices else None
                lo, hi, _ = env.action_bounds(room_index, lower_price)
                state = env.room_state(room_index)
                learner.update_ready(room_index, state, lo, hi)
                if behavior_actions is None:
                    action = learner.choose_action(state, lo, hi, epsilon)
                else:
                    action = int(behavior_actions[room_index])
                level, price, _, _ = env.resolve_room_action(room_index, action, lower_price)
                actions[room_index] = level
                states.append(state)
                decided_prices.append(price)

            _, reward, terminated, truncated, info = env.step(actions)
            episode_return += reward
            for room_index, state in enumerate(states):
                learner.record(
                    room_index,
                    state,
                    int(actions[room_index]),
                    info["per_room_rewards"][room_index] - cancellation_rewards[room_index],
                )
            if terminated or truncated:
                if penalty_rate > 0:
                    penalties = occupancy_penalties(
                        env.rooms_left, hotel.capacity, target_occupancy, penalty_rate
                    )
                    for room_index, penalty in enumerate(penalties):
                        learner.add_reward_to_last_transition(room_index, -float(penalty))
                    occupancy_penalty = float(penalties.sum())
                learner.finish_episode()
                break

        normalization = hotel.reference_price_thb * hotel.capacity
        episode_records.append(
            {
                "episode": episode_index,
                "tier": tier_name,
                "hotel_id": hotel.hotel_id,
                "mode": mode,
                "algorithm": algorithm,
                "season": season,
                "is_weekend": is_weekend,
                "epsilon": epsilon,
                "net_profit": episode_return,
                "normalized_profit": episode_return / normalization,
                "occupancy_penalty": occupancy_penalty,
                "capacity": hotel.capacity,
                "reference_price_thb": hotel.reference_price_thb,
            }
        )
        progress_interval = config["rl"]["progress_interval_episodes"]
        if (episode_index + 1) % progress_interval == 0 or episode_index + 1 == total_episodes:
            print(
                f"{algorithm}/{mode}: episode {episode_index + 1}/{total_episodes}; "
                f"{episode_index + 1} global episodes completed",
                flush=True,
            )

        checkpoint_interval = config["rl"]["checkpoint_interval_episodes"]
        if checkpoint_interval > 0 and (episode_index + 1) % checkpoint_interval == 0:
            learner.save(
                model_dir / f"q_table_{algorithm}_{mode}_checkpoint.npy",
                model_dir / f"q_counts_{algorithm}_{mode}_checkpoint.npy",
            )
            pd.DataFrame(episode_records).to_csv(
                results_dir / f"learning_curve_{algorithm}_{mode}_checkpoint.csv",
                index=False,
            )

    episode_table = pd.DataFrame(episode_records)
    window = config["rl"]["learning_curve_moving_average_window"]
    episode_table["moving_average_normalized_profit"] = (
        episode_table.groupby("tier", sort=False)["normalized_profit"]
        .transform(lambda values: values.rolling(window=window, min_periods=1).mean())
    )

    variant_name = f"q_table_{algorithm}_{mode}.npy"
    variant_path = model_dir / variant_name
    learner.save(variant_path, model_dir / f"q_counts_{algorithm}_{mode}.npy")
    if algorithm == "n_step" and mode == config["synthetic_hotels"]["domain_randomization_mode"]:
        learner.save(model_dir / "q_table.npy", model_dir / "q_counts.npy")

    episode_table.to_csv(results_dir / f"learning_curve_{algorithm}_{mode}.csv", index=False)
    if algorithm == "n_step" and mode == config["synthetic_hotels"]["domain_randomization_mode"]:
        episode_table.to_csv(results_dir / "learning_curve.csv", index=False)
    return learner, episode_table, variant_path


def train_suite(
    config: dict[str, Any] | None = None,
    episodes_per_tier: int | None = None,
) -> dict[tuple[str, str], tuple[TabularQLearner, pd.DataFrame, Path]]:
    """Train n-step and Monte Carlo tables with and without domain randomization."""
    config = config or load_config()
    runs = {}
    for algorithm in ["n_step", "monte_carlo"]:
        for mode in [
            config["synthetic_hotels"]["single_profile_mode"],
            config["synthetic_hotels"]["domain_randomization_mode"],
        ]:
            runs[(algorithm, mode)] = train_agent(
                config=config,
                algorithm=algorithm,
                mode=mode,
                episodes_per_tier=episodes_per_tier,
            )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["n_step", "monte_carlo"], default=None)
    parser.add_argument("--mode", choices=["single", "domain_randomization"], default=None)
    parser.add_argument("--episodes-per-tier", type=int, default=None)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--config", default="config/sim_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.suite:
        runs = train_suite(config, episodes_per_tier=args.episodes_per_tier)
        for key, (_, _, model_path) in runs.items():
            print(f"Saved {key[0]}/{key[1]}: {model_path}")
    else:
        algorithm = args.algorithm or "n_step"
        mode = args.mode or config["synthetic_hotels"]["domain_randomization_mode"]
        _, table, model_path = train_agent(
            config,
            algorithm=algorithm,
            mode=mode,
            episodes_per_tier=args.episodes_per_tier,
        )
        print(table.groupby("tier")["normalized_profit"].mean().to_string())
        print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
