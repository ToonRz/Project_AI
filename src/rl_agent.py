"""Masked tabular n-step Q-learning and Monte Carlo control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.env import HotelPricingEnv


TabularState = tuple[int, ...]


class TabularQLearner:
    """One shared Q-table indexed by tier, room, season, and booking state."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        algorithm: str = "n_step",
        q_table: np.ndarray | None = None,
        update_counts: np.ndarray | None = None,
    ):
        self.config = config or load_config()
        if algorithm not in {"n_step", "monte_carlo"}:
            raise ValueError(f"Unsupported tabular algorithm: {algorithm}")
        self.algorithm = algorithm
        state_config = self.config["rl"]["state"]
        room_count = len(next(iter(self.config["tiers"].values()))["rooms"])
        shape = (
            len(self.config["tiers"]),
            room_count,
            len(self.config["episode"]["season_names"]),
            2,
            len(state_config["day_bucket_ranges"]),
            state_config["remaining_bucket_count"] + 1,
            state_config["clock_periods_per_day"],
            *((self.config["market"]["price_bucket_count"],) if self.config["market"]["enabled"] else ()),
            len(self.config["pricing"]["markup_levels"]),
        )
        self.q_table = (
            np.full(
                shape,
                self.config["rl"]["q_default_value"],
                dtype=np.float64,
            )
            if q_table is None
            else q_table
        )
        if self.q_table.shape != shape:
            raise ValueError(f"Q-table shape {self.q_table.shape} does not match configured shape {shape}")
        if update_counts is None:
            if q_table is None:
                self.update_counts = np.zeros(shape, dtype=np.uint32)
            else:
                default_value = self.config["rl"]["q_default_value"]
                self.update_counts = (self.q_table != default_value).astype(np.uint32)
        else:
            self.update_counts = np.asarray(update_counts, dtype=np.uint32)
        if self.update_counts.shape != shape:
            raise ValueError(
                f"Update-count table shape {self.update_counts.shape} does not match configured shape {shape}"
            )
        self._episode_buffers: dict[int, list[dict[str, Any]]] = {}
        self.n_step = self.config["rl"]["n_step"]
        self.gamma = self.config["rl"]["gamma"]
        self.rng = np.random.default_rng(self.config["project"]["random_seed"])

    def begin_episode(self) -> None:
        room_count = len(next(iter(self.config["tiers"].values()))["rooms"])
        self._episode_buffers = {room_index: [] for room_index in range(room_count)}

    def epsilon(self, global_episode_index: int, total_episodes: int, start: float = 1.0) -> float:
        decay_fraction = self.config["rl"]["epsilon_decay_episode_fraction"]
        minimum = self.config["rl"]["epsilon_min"]
        return max(minimum, start * (1.0 - global_episode_index / (decay_fraction * total_episodes)))

    def choose_action(
        self,
        state: TabularState,
        lo: int,
        hi: int,
        epsilon: float,
        rng: np.random.Generator | None = None,
    ) -> int:
        rng = rng or self.rng
        allowed = np.arange(lo, hi + 1, dtype=np.int64)
        if rng.random() < epsilon:
            return int(rng.choice(allowed))
        q_values = self.q_table[state + (allowed,)]
        best_value = q_values.max()
        best_actions = allowed[np.isclose(q_values, best_value, rtol=1e-12, atol=1e-12)]
        return int(rng.choice(best_actions))

    def _update(self, state: TabularState, action: int, target: float) -> None:
        key = state + (int(action),)
        self.update_counts[key] += 1
        count = int(self.update_counts[key])
        if self.algorithm == "n_step":
            minimum = self.config["rl"]["alpha_n_step_min"]
            exponent = self.config["rl"]["alpha_n_step_count_exponent"]
            alpha = max(minimum, 1.0 / (count**exponent))
        else:
            minimum = self.config["rl"]["alpha_monte_carlo_min"]
            alpha = max(minimum, 1.0 / count)
        self.q_table[key] += alpha * (target - self.q_table[key])

    def record(self, room_index: int, state: TabularState, action: int, reward: float) -> None:
        self._episode_buffers[room_index].append(
            {"state": state, "action": int(action), "reward": float(reward)}
        )

    def add_reward_to_last_transition(self, room_index: int, reward: float) -> None:
        """Attach a reward arriving before the next action to its prior transition."""
        if reward == 0.0:
            return
        buffer = self._episode_buffers[room_index]
        if not buffer:
            raise RuntimeError("Cannot attach a delayed reward before the room has an action.")
        buffer[-1]["reward"] += float(reward)

    def update_ready(
        self,
        room_index: int,
        next_state: TabularState,
        lo: int,
        hi: int,
    ) -> None:
        """Apply one n-step update using only actions allowed in the next-state mask."""
        if self.algorithm != "n_step":
            return
        buffer = self._episode_buffers[room_index]
        if len(buffer) < self.n_step:
            return

        target = sum(
            self.gamma**index * transition["reward"]
            for index, transition in enumerate(buffer[: self.n_step])
        )
        allowed_next_values = self.q_table[next_state + (slice(lo, hi + 1),)]
        target += self.gamma**self.n_step * float(allowed_next_values.max())
        transition = buffer.pop(0)
        self._update(transition["state"], transition["action"], target)

    def finish_episode(self) -> None:
        """Flush remaining returns at terminal without bootstrapping."""
        for buffer in self._episode_buffers.values():
            if self.algorithm == "n_step":
                while buffer:
                    target = 0.0
                    for index, transition in enumerate(buffer):
                        target += self.gamma**index * transition["reward"]
                    first = buffer.pop(0)
                    self._update(first["state"], first["action"], target)
            else:
                return_value = 0.0
                for transition in reversed(buffer):
                    return_value = transition["reward"] + self.gamma * return_value
                    self._update(transition["state"], transition["action"], return_value)
                buffer.clear()
        self._episode_buffers = {}

    def save(self, path: str | Path, counts_path: str | Path | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)
        if counts_path is not None:
            counts_path = Path(counts_path)
            counts_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(counts_path, self.update_counts)


class QTablePolicy:
    """Greedy environment policy with per-room sequential mask handling."""

    name = "RL"

    def __init__(
        self,
        learner: TabularQLearner,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
    ):
        self.learner = learner
        self.config = config or learner.config
        self.rng = np.random.default_rng(
            self.config["project"]["random_seed"] if seed is None else seed
        )

    def select_actions(self, env: HotelPricingEnv) -> np.ndarray:
        env.prepare_step()
        actions = np.zeros(len(env.levels), dtype=np.int64)
        decided_prices: list[float] = []
        for room_index in range(len(actions)):
            lower = decided_prices[-1] if decided_prices else None
            lo, hi, _ = env.action_bounds(room_index, lower)
            state = env.room_state(room_index)
            state_visits = int(self.learner.update_counts[state].sum())
            if state_visits == 0:
                target_level = self.config["rl"]["unseen_state_markup_level"]
                action = min(max(int(target_level), lo), hi)
            else:
                action = self.learner.choose_action(state, lo, hi, 0.0, self.rng)
            level, price, _, _ = env.resolve_room_action(room_index, action, lower)
            actions[room_index] = level
            decided_prices.append(price)
        return actions
