"""Stochastic customer environment with fixed-seed scenario generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.config import load_config
from src.hotels import Hotel


@dataclass(frozen=True)
class ScenarioCustomer:
    customer_id: int
    step_index: int
    segment: str
    wtp_thb: float
    via_ota: bool
    cancel_draw: float
    cancel_time_draw: float


@dataclass(frozen=True)
class CustomerScenario:
    seed: int
    customers_by_step: tuple[tuple[ScenarioCustomer, ...], ...]
    visitor_count_by_segment: dict[str, int]


@dataclass(frozen=True)
class Booking:
    customer_id: int
    segment: str
    room_index: int
    price_thb: float
    via_ota: bool
    cancel_draw: float
    cancel_time_draw: float
    wtp_thb: float


@dataclass(frozen=True)
class CustomerDecision:
    customer_id: int
    segment: str
    status: str
    room_index: int | None
    wtp_thb: float
    via_ota: bool


class CustomerAgent:
    """Generate customers once and replay the same customer scenario per policy."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

    def _daily_weight(self, hotel: Hotel, segment: str, days_remaining: int) -> float:
        weights = hotel.arrival_weights.get(segment)
        if weights and len(weights) >= days_remaining:
            return float(weights[days_remaining - 1])

        tier_segment = self.config["tiers"][hotel.tier]["segments"][segment]["hour_profile"]
        days = np.arange(1, self.config["episode"]["booking_horizon_days"] + 1, dtype=float)
        if segment == "leisure":
            center = tier_segment["center"]
            spread = tier_segment["center_sigma"]
            fallback = np.exp(-((days - center) ** 2) / (2 * spread**2))
        else:
            decay = tier_segment["arrival_decay_days"]
            fallback = np.exp(-((days - 1) / decay))
        fallback /= fallback.sum()
        return float(fallback[days_remaining - 1])

    def _hour_probabilities(self, hotel: Hotel, segment: str) -> np.ndarray:
        profile = self.config["demand"]["hourly_profiles"][segment]
        clock = np.arange(self.config["episode"]["hours_per_day"], dtype=float)

        def gaussian(spec: dict[str, float]) -> np.ndarray:
            return spec["weight"] * np.exp(
                -((clock - spec["mean"]) ** 2) / (2 * spec["sigma"] ** 2)
            )

        weights = profile["floor"] + gaussian(profile["gaussian_1"])
        weights += gaussian(profile["gaussian_2"])
        weights = np.roll(weights, hotel.h_shift)
        return weights / weights.sum()

    def _daily_mean(
        self,
        hotel: Hotel,
        segment: str,
        days_remaining: int,
        season: str,
        is_weekend: bool,
    ) -> float:
        segment_share = hotel.business_share if segment == "business" else 1.0 - hotel.business_share
        dow_name = "weekend" if is_weekend else "weekday"
        season_base = hotel.season_multipliers[segment][season]
        dow_base = hotel.dow_multipliers[dow_name][segment]
        season_factor = 1.0 + hotel.amp_season * (season_base - 1.0)
        dow_factor = 1.0 + hotel.amp_weekend * (dow_base - 1.0)
        return (
            hotel.capacity
            * hotel.demand_per_room_episode
            * segment_share
            * season_factor
            * dow_factor
            * self._daily_weight(hotel, segment, days_remaining)
        )

    def generate_scenario(
        self,
        hotel: Hotel,
        season: str,
        is_weekend: bool,
        seed: int,
    ) -> CustomerScenario:
        """Pre-sample visitors, WTP, channel, and cancellation random numbers."""
        rng = np.random.default_rng(seed)
        episode = self.config["episode"]
        days = episode["booking_horizon_days"]
        hours = episode["hours_per_day"]
        step_count = episode["steps"]
        segment_names = list(hotel.segments)
        hourly_probs = {segment: self._hour_probabilities(hotel, segment) for segment in segment_names}
        arrivals: list[list[ScenarioCustomer]] = [[] for _ in range(step_count)]
        visitor_counts = {segment: 0 for segment in segment_names}
        customer_id = 0

        for days_remaining in range(days, 0, -1):
            day_start = (days - days_remaining) * hours
            for segment in segment_names:
                mean = self._daily_mean(hotel, segment, days_remaining, season, is_weekend)
                visitor_count = int(rng.poisson(mean))
                visitor_counts[segment] += visitor_count
                clocks = rng.choice(hours, size=visitor_count, p=hourly_probs[segment])
                segment_config = hotel.segments[segment]
                ota_share = (
                    hotel.ota_share_leisure if segment == "leisure" else hotel.ota_share_business
                )
                for clock in clocks:
                    step_index = day_start + int(clock)
                    customer = ScenarioCustomer(
                        customer_id=customer_id,
                        step_index=step_index,
                        segment=segment,
                        wtp_thb=float(
                            hotel.reference_price_thb
                            * segment_config["wtp_median_reference"]
                            * np.exp(segment_config["wtp_sigma"] * rng.normal())
                        ),
                        via_ota=bool(rng.random() < ota_share),
                        cancel_draw=float(rng.random()),
                        cancel_time_draw=float(rng.random()),
                    )
                    arrivals[step_index].append(customer)
                    customer_id += 1

        for hour_customers in arrivals:
            rng.shuffle(hour_customers)
        return CustomerScenario(
            seed=seed,
            customers_by_step=tuple(tuple(customers) for customers in arrivals),
            visitor_count_by_segment=visitor_counts,
        )

    def respond(
        self,
        customers: Sequence[ScenarioCustomer],
        hotel: Hotel,
        prices: Sequence[float],
        rooms_left: np.ndarray,
    ) -> tuple[list[Booking], list[CustomerDecision]]:
        """Let each visitor select the highest-surplus available room."""
        prices_array = np.asarray(prices, dtype=float)
        available = rooms_left > 0
        sizes = np.asarray([room.size_m2 for room in hotel.rooms], dtype=float)
        relative_sizes = sizes / sizes.min()
        bookings: list[Booking] = []
        decisions: list[CustomerDecision] = []

        for customer in customers:
            if not available.any():
                decisions.append(
                    CustomerDecision(
                        customer.customer_id,
                        customer.segment,
                        "full",
                        None,
                        customer.wtp_thb,
                        customer.via_ota,
                    )
                )
                continue

            beta = hotel.segments[customer.segment]["size_beta"]
            surplus = customer.wtp_thb * np.power(relative_sizes, beta) - prices_array
            surplus = np.where(available, surplus, -np.inf)
            room_index = int(np.argmax(surplus))
            if surplus[room_index] < 0.0:
                decisions.append(
                    CustomerDecision(
                        customer.customer_id,
                        customer.segment,
                        "too_expensive",
                        None,
                        customer.wtp_thb,
                        customer.via_ota,
                    )
                )
                continue

            rooms_left[room_index] -= 1
            available[room_index] = rooms_left[room_index] > 0
            booking = Booking(
                customer_id=customer.customer_id,
                segment=customer.segment,
                room_index=room_index,
                price_thb=float(prices_array[room_index]),
                via_ota=customer.via_ota,
                cancel_draw=customer.cancel_draw,
                cancel_time_draw=customer.cancel_time_draw,
                wtp_thb=customer.wtp_thb,
            )
            bookings.append(booking)
            decisions.append(
                CustomerDecision(
                    customer.customer_id,
                    customer.segment,
                    "booked",
                    room_index,
                    customer.wtp_thb,
                    customer.via_ota,
                )
            )
        return bookings, decisions
