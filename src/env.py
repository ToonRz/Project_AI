"""Gymnasium environment for hourly hotel-room dynamic pricing."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import PROJECT_ROOT, load_config
from src.customer_agent import Booking, CustomerAgent, CustomerDecision, CustomerScenario
from src.hotels import Hotel, generate_hotel_population

if TYPE_CHECKING:
    from src.market import Market


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@dataclass
class Reservation:
    booking: Booking
    net_profit: float
    booked_at_hours_remaining: int
    cancellation_at_hours_remaining: int | None = None


class HotelPricingEnv(gym.Env):
    """One 30-day sale horizon with a per-room hourly action and shared inventory."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        hotel: Hotel | None = None,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config or load_config()
        self.render_mode = render_mode
        self.customer_agent = CustomerAgent(self.config)
        tier_names = list(self.config["tiers"])
        action_count = len(self.config["pricing"]["markup_levels"])
        room_count = len(self.config["tiers"][tier_names[0]]["rooms"])
        self.action_space = spaces.MultiDiscrete(np.full(room_count, action_count, dtype=np.int64))
        observation_size = len(tier_names) + len(self.config["episode"]["season_names"]) + 1 + 1 + room_count + 2 + room_count
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )

        train, _ = generate_hotel_population(
            self.config, mode=self.config["synthetic_hotels"]["single_profile_mode"]
        )
        self._base_hotels: dict[str, Hotel] = {tier: hotels[0] for tier, hotels in train.items()}
        if hotel is None:
            hotel = self._base_hotels[tier_names[0]]
        self.default_hotel = hotel
        self.hotel = hotel
        self.room_reference_prices = self._room_reference_prices(hotel)
        self.price_levels = np.asarray(self.config["pricing"]["markup_levels"], dtype=float)
        self.season = self.config["episode"]["season_names"][0]
        self.is_weekend = False
        self.scenario_seed = 0
        self.scenario: CustomerScenario | None = None
        self.market: Market | None = None
        self.step_index = 0
        self.hours_remaining = self.config["episode"]["steps"]
        self.levels = np.full(room_count, self.config["pricing"]["initial_price_level"], dtype=np.int64)
        self.prices = np.zeros(room_count, dtype=float)
        self.rooms_left = np.zeros(room_count, dtype=np.int64)
        self._cancel_queue: dict[int, list[Reservation]] = defaultdict(list)
        self._active_reservations: dict[int, Reservation] = {}
        self._episode_reward = 0.0
        self._gross_revenue = 0.0
        self._refunds = 0.0
        self._total_commission = 0.0
        self._booking_count = 0
        self._cancel_count = 0
        self._last_action_masks: list[list[int]] = []
        self._last_decisions: list[CustomerDecision] = []
        self._hour_prepared = False
        self._prepared_reward_by_room = np.zeros(room_count, dtype=float)
        self._prepared_cancellations: list[dict[str, Any]] = []

    def _room_reference_prices(self, hotel: Hotel) -> np.ndarray:
        pricing = self.config["pricing"]
        reference_size = min(room.size_m2 for room in hotel.rooms)
        unit = pricing["reference_rounding_thb"]
        return np.asarray(
            [
                int(round(hotel.reference_price_thb * (room.size_m2 / reference_size) ** pricing["reference_size_elasticity"] / unit) * unit)
                for room in hotel.rooms
            ],
            dtype=float,
        )

    def price_for_level(self, room_index: int, level: int) -> float:
        unit = self.config["pricing"]["price_rounding_thb"]
        return float(int(round(self.room_reference_prices[room_index] * self.price_levels[level] / unit) * unit))

    def _ceiling_price(self, value: float) -> float:
        unit = self.config["pricing"]["price_rounding_thb"]
        return float(math.ceil(value / unit) * unit)

    def get_legal_levels(
        self, room_index: int, lower_room_price: float | None = None
    ) -> tuple[list[int], bool]:
        """Return allowed grid levels and whether the large-room rule forces a price."""
        action_count = len(self.price_levels)
        if self.step_index == 0:
            candidates = list(range(action_count))
        else:
            max_change = self.config["pricing"]["max_level_change_per_hour"]
            previous = int(self.levels[room_index])
            candidates = list(
                range(max(0, previous - max_change), min(action_count - 1, previous + max_change) + 1)
            )

        if room_index > 0 and lower_room_price is not None:
            minimum_ratio = self.config["pricing"]["minimum_large_room_price_ratio"]
            candidates = [
                level
                for level in candidates
                if self.price_for_level(room_index, level) + 1e-9 >= minimum_ratio * lower_room_price
            ]
            if not candidates:
                return [action_count - 1], True
        return candidates, False

    def action_bounds(
        self, room_index: int, lower_room_price: float | None = None
    ) -> tuple[int, int, bool]:
        allowed, forced = self.get_legal_levels(room_index, lower_room_price)
        return min(allowed), max(allowed), forced

    def action_masks(self) -> np.ndarray:
        """MaskablePPO-compatible flat mask based on the last published room prices."""
        masks = []
        for room_index in range(len(self.levels)):
            lower = float(self.prices[room_index - 1]) if room_index > 0 else None
            allowed, _ = self.get_legal_levels(room_index, lower)
            mask = np.zeros(len(self.price_levels), dtype=bool)
            mask[allowed] = True
            masks.append(mask)
        return np.concatenate(masks)

    def resolve_room_action(
        self,
        room_index: int,
        requested_level: int,
        lower_room_price: float | None = None,
    ) -> tuple[int, float, list[int], bool]:
        allowed, forced = self.get_legal_levels(room_index, lower_room_price)
        if requested_level not in allowed:
            requested_level = min(allowed, key=lambda level: abs(level - requested_level))
        if forced:
            minimum_ratio = self.config["pricing"]["minimum_large_room_price_ratio"]
            price = self._ceiling_price(minimum_ratio * float(lower_room_price))
        else:
            price = self.price_for_level(room_index, requested_level)
        return int(requested_level), float(price), allowed, forced

    def room_state(self, room_index: int) -> tuple[int, ...]:
        """Return the tabular state for a room before its action is selected."""
        tier_id = self.config["tiers"][self.hotel.tier]["id"]
        season_id = self.config["episode"]["season_names"].index(self.season)
        hours = self.config["episode"]["hours_per_day"]
        h = max(1, self.config["episode"]["steps"] - self.step_index)
        days_remaining = (h - 1) // hours + 1
        day_bucket = next(
            index
            for index, (low, high) in enumerate(self.config["rl"]["state"]["day_bucket_ranges"])
            if low <= days_remaining <= high
        )
        room_capacity = self.hotel.rooms[room_index].count
        remaining_buckets = self.config["rl"]["state"]["remaining_bucket_count"]
        remaining_bucket = int(math.ceil(self.rooms_left[room_index] / room_capacity * remaining_buckets))
        clock = self.step_index % hours
        period = clock // self.config["rl"]["state"]["time_period_hours"]
        state = (
            int(tier_id),
            int(room_index),
            int(season_id),
            int(self.is_weekend),
            int(day_bucket),
            int(remaining_bucket),
            int(period),
        )
        if self.config["market"]["enabled"]:
            return state + (self._market_bucket(room_index),)
        return state

    def _market_bucket(self, room_index: int) -> int:
        """Our previous-hour price vs the competitor median: 0 cheaper, 1 within band, 2 dearer."""
        median = (
            self.market.median_price(max(self.step_index - 1, 0), room_index)
            if self.market is not None
            else None
        )
        if median is None:
            return 1
        our = float(self.prices[room_index])
        band = self.config["market"]["price_band"]
        tolerance = 1e-9 * median
        if our < (1.0 - band) * median - tolerance:
            return 0
        if our > (1.0 + band) * median + tolerance:
            return 2
        return 1

    def _observation(self) -> np.ndarray:
        tier_names = list(self.config["tiers"])
        season_names = self.config["episode"]["season_names"]
        tier_one_hot = np.eye(len(tier_names), dtype=np.float32)[tier_names.index(self.hotel.tier)]
        season_one_hot = np.eye(len(season_names), dtype=np.float32)[season_names.index(self.season)]
        hours = self.config["episode"]["hours_per_day"]
        h = max(0, self.config["episode"]["steps"] - self.step_index)
        clock = self.step_index % hours
        remaining = self.rooms_left / np.asarray([room.count for room in self.hotel.rooms], dtype=float)
        time_features = np.asarray(
            [np.sin(2 * np.pi * clock / hours), np.cos(2 * np.pi * clock / hours)],
            dtype=np.float32,
        )
        level_features = self.levels.astype(np.float32) / (len(self.price_levels) - 1)
        return np.concatenate(
            [
                tier_one_hot,
                season_one_hot,
                np.asarray([float(self.is_weekend), h / self.config["episode"]["steps"]], dtype=np.float32),
                remaining.astype(np.float32),
                time_features,
                level_features,
            ]
        ).astype(np.float32)

    def _reset_accounting(self) -> None:
        room_count = len(self.hotel.rooms)
        self.step_index = 0
        self.hours_remaining = self.config["episode"]["steps"]
        self.levels = np.full(
            room_count, self.config["pricing"]["initial_price_level"], dtype=np.int64
        )
        self.prices = np.asarray(
            [self.price_for_level(index, self.levels[index]) for index in range(room_count)],
            dtype=float,
        )
        self.rooms_left = np.asarray([room.count for room in self.hotel.rooms], dtype=np.int64)
        self._cancel_queue = defaultdict(list)
        self._active_reservations = {}
        self._episode_reward = 0.0
        self._gross_revenue = 0.0
        self._refunds = 0.0
        self._total_commission = 0.0
        self._booking_count = 0
        self._cancel_count = 0
        self._last_action_masks = []
        self._last_decisions = []
        self._hour_prepared = False
        self._prepared_reward_by_room = np.zeros(room_count, dtype=float)
        self._prepared_cancellations = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "hotel" in options:
            self.hotel = options["hotel"]
        elif "tier" in options and options["tier"] in self._base_hotels:
            self.hotel = self._base_hotels[options["tier"]]
        elif self.default_hotel is not None:
            self.hotel = self.default_hotel
        self.room_reference_prices = self._room_reference_prices(self.hotel)
        season_names = self.config["episode"]["season_names"]
        season_probabilities = self.config["episode"]["season_probabilities"]
        if "season" in options:
            self.season = str(options["season"])
        else:
            self.season = str(self.np_random.choice(season_names, p=season_probabilities))
        if "is_weekend" in options:
            self.is_weekend = bool(options["is_weekend"])
        else:
            self.is_weekend = bool(
                self.np_random.random() < self.config["episode"]["weekend_probability"]
            )
        self.market = options.get("market")
        scenario_seed = options.get("scenario_seed", seed)
        if scenario_seed is None:
            scenario_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        self.scenario_seed = int(scenario_seed)
        self.scenario = self.customer_agent.generate_scenario(
            self.hotel, self.season, self.is_weekend, self.scenario_seed
        )
        self._reset_accounting()
        return self._observation(), {
            "hotel_id": self.hotel.hotel_id,
            "tier": self.hotel.tier,
            "season": self.season,
            "is_weekend": self.is_weekend,
            "scenario_seed": self.scenario_seed,
        }

    def _cancel_reservation(self, reservation: Reservation, reward_by_room: np.ndarray, h: int) -> dict[str, Any]:
        booking = reservation.booking
        room_index = booking.room_index
        self.rooms_left[room_index] += 1
        reward_by_room[room_index] -= reservation.net_profit
        self._refunds += booking.price_thb
        self._cancel_count += 1
        self._active_reservations.pop(booking.customer_id, None)
        return {
            "customer_id": booking.customer_id,
            "room_index": room_index,
            "price_thb": booking.price_thb,
            "net_profit_reversed": reservation.net_profit,
            "hours_remaining": h,
        }

    def prepare_step(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Process cancellations due now before exposing states and action masks."""
        if self.scenario is None:
            raise RuntimeError("Call reset() before preparing a step.")
        if self.step_index >= self.config["episode"]["steps"]:
            raise RuntimeError("Episode is complete; call reset() before preparing another step.")
        if self._hour_prepared:
            return self._prepared_reward_by_room.copy(), list(self._prepared_cancellations)

        h = self.config["episode"]["steps"] - self.step_index
        room_count = len(self.hotel.rooms)
        reward_by_room = np.zeros(room_count, dtype=float)
        cancellations = [
            self._cancel_reservation(reservation, reward_by_room, h)
            for reservation in self._cancel_queue.pop(h, [])
        ]
        self._prepared_reward_by_room = reward_by_room
        self._prepared_cancellations = cancellations
        self._hour_prepared = True
        return reward_by_room.copy(), list(cancellations)

    def step(self, actions: np.ndarray | list[int]):
        if self.scenario is None:
            raise RuntimeError("Call reset() before step().")
        if self.step_index >= self.config["episode"]["steps"]:
            raise RuntimeError("Episode is complete; call reset() before stepping again.")

        action_values = np.asarray(actions, dtype=int)
        if action_values.shape != self.levels.shape:
            raise ValueError(f"Expected {self.levels.size} room actions; got shape {action_values.shape}")
        action_count = len(self.price_levels)
        action_values = np.clip(action_values, 0, action_count - 1)

        self.prepare_step()
        h = self.config["episode"]["steps"] - self.step_index
        days_remaining = (h - 1) // self.config["episode"]["hours_per_day"] + 1
        room_count = len(self.hotel.rooms)
        reward_by_room = self._prepared_reward_by_room.copy()
        cancellations = list(self._prepared_cancellations)

        new_levels = np.empty(room_count, dtype=np.int64)
        new_prices = np.empty(room_count, dtype=float)
        masks: list[list[int]] = []
        forced_prices: list[bool] = []
        for room_index, requested in enumerate(action_values):
            lower_price = float(new_prices[room_index - 1]) if room_index > 0 else None
            level, price, allowed, forced = self.resolve_room_action(
                room_index, int(requested), lower_price
            )
            new_levels[room_index] = level
            new_prices[room_index] = price
            masks.append(allowed)
            forced_prices.append(forced)
        self.levels = new_levels
        self.prices = new_prices

        customers = self.scenario.customers_by_step[self.step_index]
        bookings, decisions = self.customer_agent.respond(
            customers,
            self.hotel,
            self.prices,
            self.rooms_left,
            self.market.offers(self.step_index) if self.market is not None else None,
        )
        self._last_decisions = decisions
        commission_rate = self.config["pricing"]["ota_commission"]
        cost = self.hotel.variable_cost_thb
        immediate_cancellations: list[Reservation] = []
        cancellation_start = self.config["demand"]["cancellation_lead_factor_start"]
        cancellation_slope = self.config["demand"]["cancellation_lead_factor_slope"]
        horizon = self.config["episode"]["booking_horizon_days"]
        for booking in bookings:
            commission = commission_rate if booking.via_ota else 0.0
            net_profit = booking.price_thb * (1.0 - commission) - cost
            reward_by_room[booking.room_index] += net_profit
            self._gross_revenue += booking.price_thb
            self._total_commission += booking.price_thb * commission
            self._booking_count += 1
            probability = (
                self.hotel.segments[booking.segment]["cancel_base"]
                * (cancellation_start + cancellation_slope * days_remaining / horizon)
            )
            if booking.cancel_draw < probability:
                cancellation_remaining = 1 + int(
                    np.floor(booking.cancel_time_draw * (h - 1))
                )
                reservation = Reservation(
                    booking=booking,
                    net_profit=net_profit,
                    booked_at_hours_remaining=h,
                    cancellation_at_hours_remaining=cancellation_remaining,
                )
                self._active_reservations[booking.customer_id] = reservation
                if cancellation_remaining == h:
                    immediate_cancellations.append(reservation)
                else:
                    self._cancel_queue[cancellation_remaining].append(reservation)
            else:
                self._active_reservations[booking.customer_id] = Reservation(
                    booking=booking,
                    net_profit=net_profit,
                    booked_at_hours_remaining=h,
                )

        for reservation in immediate_cancellations:
            cancellations.append(self._cancel_reservation(reservation, reward_by_room, h))

        self.step_index += 1
        self.hours_remaining = self.config["episode"]["steps"] - self.step_index
        self._hour_prepared = False
        self._prepared_reward_by_room = np.zeros(room_count, dtype=float)
        self._prepared_cancellations = []
        reward = float(reward_by_room.sum())
        self._episode_reward += reward
        terminated = self.step_index >= self.config["episode"]["steps"]
        info: dict[str, Any] = {
            "h": h,
            "days_remaining": days_remaining,
            "room_states": [self.room_state(index) for index in range(room_count)] if not terminated else [],
            "action_levels": self.levels.tolist(),
            "action_masks": masks,
            "forced_large_room_prices": forced_prices,
            "prices": self.prices.tolist(),
            "rooms_left": self.rooms_left.tolist(),
            "per_room_rewards": reward_by_room.tolist(),
            "bookings": [self._booking_log(booking) for booking in bookings],
            "cancellations": cancellations,
            "customer_log": [self._decision_log(decision) for decision in decisions],
        }
        if terminated:
            info["episode_metrics"] = self.episode_metrics()
            info["episode_metrics"]["reward_sum"] = self._episode_reward
        return self._observation(), reward, terminated, False, info

    def _booking_log(self, booking: Booking) -> dict[str, Any]:
        return {
            "customer_id": booking.customer_id,
            "segment": booking.segment,
            "room_index": booking.room_index,
            "room_name": self.hotel.rooms[booking.room_index].name,
            "price_thb": booking.price_thb,
            "via_ota": booking.via_ota,
        }

    def _decision_log(self, decision: CustomerDecision) -> dict[str, Any]:
        return {
            "customer_id": decision.customer_id,
            "segment": decision.segment,
            "status": decision.status,
            "room_index": decision.room_index,
            "room_name": (
                self.hotel.rooms[decision.room_index].name
                if decision.room_index is not None
                else None
            ),
            "wtp_thb": decision.wtp_thb,
            "via_ota": decision.via_ota,
            "competitor_index": decision.competitor_index,
        }

    def episode_metrics(self) -> dict[str, float | int | str | bool]:
        capacity = self.hotel.capacity
        active_rooms = capacity - int(self.rooms_left.sum())
        revenue = self._gross_revenue - self._refunds
        return {
            "hotel_id": self.hotel.hotel_id,
            "tier": self.hotel.tier,
            "season": self.season,
            "is_weekend": self.is_weekend,
            "gross_revenue": self._gross_revenue,
            "refunds": self._refunds,
            "revenue": revenue,
            "net_profit": self._episode_reward,
            "occupancy": active_rooms / capacity,
            "adr": revenue / active_rooms if active_rooms else 0.0,
            "revpar": revenue / capacity,
            "profit_per_available_room": self._episode_reward / capacity,
            "rooms_available": capacity,
            "rooms_sold": active_rooms,
            "booking_count": self._booking_count,
            "cancellation_count": self._cancel_count,
            "ota_commission": self._total_commission,
        }

    def render(self):
        return None


def make_default_env(tier: str = "budget", config: dict[str, Any] | None = None) -> HotelPricingEnv:
    config = config or load_config()
    train, _ = generate_hotel_population(config, mode=config["synthetic_hotels"]["single_profile_mode"])
    return HotelPricingEnv(hotel=train[tier][0], config=config)
