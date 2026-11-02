"""Cleaning, segmentation, and stay-night expansion for Hotel Booking Demand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_bookings(config: dict[str, Any]) -> pd.DataFrame:
    """Read the raw booking CSV from the configured path."""
    return pd.read_csv(_project_path(config["paths"]["hotel_bookings"]))


def clean_and_segment_bookings(
    bookings: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Apply the requested ADR filter and add segment/date/channel fields."""
    data = bookings.copy()
    data["adr"] = pd.to_numeric(data["adr"], errors="coerce")
    minimum_adr = config["data"]["minimum_adr"]
    positive_adr = data.loc[data["adr"] > minimum_adr, "adr"]
    adr_limit = positive_adr.quantile(config["data"]["adr_trim_quantile"])
    data = data.loc[data["adr"].gt(minimum_adr) & data["adr"].le(adr_limit)].copy()

    business_segments = set(config["data"]["business_market_segments"])
    data["segment"] = np.where(
        data["market_segment"].isin(business_segments), "business", "leisure"
    )
    ota_channels = set(config["data"]["ota_distribution_channels"])
    data["is_ota"] = data["distribution_channel"].isin(ota_channels)

    date_columns = config["data"]["arrival_date_columns"]
    month_order = config["eda"]["month_order"]
    data["arrival_month_number"] = data[date_columns["month"]].map(
        {month: number for number, month in enumerate(month_order, start=1)}
    )
    data["arrival_date"] = pd.to_datetime(
        {
            "year": data[date_columns["year"]],
            "month": data["arrival_month_number"],
            "day": data[date_columns["day"]],
        },
        errors="coerce",
    )
    data["arrival_weekday"] = data["arrival_date"].dt.day_name()

    stay_columns = config["data"]["stay_length_columns"]
    data["stay_nights"] = (
        data[stay_columns["weekday_nights"]].fillna(0)
        + data[stay_columns["weekend_nights"]].fillna(0)
    ).astype(int)

    edges = [float(edge) if edge != ".inf" else np.inf for edge in config["data"]["lead_time_bins"]["edges"]]
    data["lead_time_bucket"] = pd.cut(
        data["lead_time"],
        bins=edges,
        labels=config["data"]["lead_time_bins"]["labels"],
        include_lowest=True,
    )
    return data.reset_index(drop=True)


def expand_stay_nights(bookings: pd.DataFrame) -> pd.DataFrame:
    """Expand realized (non-cancelled) multi-night stays into one row per night."""
    stays = bookings.loc[
        bookings["is_canceled"].eq(0)
        & bookings["arrival_date"].notna()
        & bookings["stay_nights"].gt(0)
    ].copy()
    stays["stay_night_date"] = stays.apply(
        lambda row: pd.date_range(row["arrival_date"], periods=row["stay_nights"]),
        axis=1,
    )
    expanded = stays.explode("stay_night_date", ignore_index=True)
    expanded["stay_night_date"] = pd.to_datetime(expanded["stay_night_date"])
    return expanded


def segment_summary(bookings: pd.DataFrame) -> pd.DataFrame:
    """Summarize booking behavior for the leisure and business segments."""
    summary = (
        bookings.groupby("segment", observed=False)
        .agg(
            bookings=("segment", "size"),
            mean_lead_time_days=("lead_time", "mean"),
            median_lead_time_days=("lead_time", "median"),
            mean_adr=("adr", "mean"),
            median_adr=("adr", "median"),
            cancel_rate=("is_canceled", "mean"),
            ota_share=("is_ota", "mean"),
        )
        .reset_index()
    )
    summary["booking_share"] = summary["bookings"] / summary["bookings"].sum()
    return summary


def run_preprocessing(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, clean, summarize, and persist booking-level and stay-night data."""
    config = config or load_config()
    raw = load_bookings(config)
    cleaned = clean_and_segment_bookings(raw, config)
    stay_nights = expand_stay_nights(cleaned)
    summary = segment_summary(cleaned)

    raw_path = _project_path(config["paths"]["hotel_bookings"])
    cleaned.to_csv(raw_path.with_name("processed_bookings.csv"), index=False)
    stay_nights.to_csv(raw_path.with_name("stay_nights.csv"), index=False)
    output = _project_path(config["paths"]["eda_results"])
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "segment_summary.csv", index=False)
    return cleaned, stay_nights, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sim_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    cleaned, stay_nights, summary = run_preprocessing(config)
    print(f"Cleaned bookings: {len(cleaned):,}")
    print(f"Expanded stay nights: {len(stay_nights):,}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
