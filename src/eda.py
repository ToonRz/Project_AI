"""Generate reproducible exploratory plots from cleaned booking records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data_prep import run_preprocessing


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def create_eda_plots(
    bookings: pd.DataFrame, output_dir: str | Path, config: dict[str, Any]
) -> list[Path]:
    """Write the five requested EDA charts and return their paths."""
    output = _project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    eda_config = config["eda"]
    size = tuple(eda_config["figure_size_inches"])
    dpi = eda_config["figure_dpi"]
    saved: list[Path] = []

    figure, axis = plt.subplots(figsize=size)
    for segment in ["leisure", "business"]:
        axis.hist(
            bookings.loc[bookings["segment"].eq(segment), "lead_time"],
            bins=eda_config["histogram_bins"],
            alpha=0.55,
            label=segment.title(),
            density=True,
        )
    axis.set(title="Lead time by customer segment", xlabel="Days before arrival", ylabel="Density")
    axis.legend()
    saved.append(_save_figure(figure, output / "lead_time_by_segment.png", dpi))

    monthly = (
        bookings.groupby("arrival_month_number", observed=False)["adr"]
        .median()
        .reindex(range(1, len(eda_config["month_order"]) + 1))
    )
    figure, axis = plt.subplots(figsize=size)
    axis.plot(monthly.index, monthly.values, marker="o")
    axis.set(
        title="Median ADR by arrival month",
        xlabel="Arrival month",
        ylabel="ADR",
        xticks=range(1, len(eda_config["month_order"]) + 1),
        xticklabels=[month[:3] for month in eda_config["month_order"]],
    )
    axis.tick_params(axis="x", rotation=45)
    saved.append(_save_figure(figure, output / "adr_by_month.png", dpi))

    weekday_adr = (
        bookings.groupby("arrival_weekday", observed=False)["adr"]
        .median()
        .reindex(eda_config["weekday_order"])
    )
    figure, axis = plt.subplots(figsize=size)
    axis.bar(weekday_adr.index, weekday_adr.values)
    axis.set(title="Median ADR by arrival weekday", xlabel="Weekday", ylabel="ADR")
    axis.tick_params(axis="x", rotation=35)
    saved.append(_save_figure(figure, output / "adr_by_weekday.png", dpi))

    cancel_by_lead = (
        bookings.groupby("lead_time_bucket", observed=False)["is_canceled"]
        .mean()
        .reindex(config["data"]["lead_time_bins"]["labels"])
    )
    figure, axis = plt.subplots(figsize=size)
    axis.bar(cancel_by_lead.index.astype(str), cancel_by_lead.values)
    axis.set(
        title="Cancellation rate by lead-time bucket",
        xlabel="Days before arrival",
        ylabel="Cancellation rate",
        ylim=(0, 1),
    )
    saved.append(_save_figure(figure, output / "cancellation_by_lead_time.png", dpi))

    ota_share = bookings.groupby("segment", observed=False)["is_ota"].mean().reindex(["leisure", "business"])
    figure, axis = plt.subplots(figsize=size)
    axis.bar([name.title() for name in ota_share.index], ota_share.values)
    axis.set(title="OTA share by customer segment (TA/TO)", xlabel="Segment", ylabel="OTA share", ylim=(0, 1))
    saved.append(_save_figure(figure, output / "ota_share_by_segment.png", dpi))
    return saved


def _save_figure(figure: plt.Figure, path: Path, dpi: int) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def run_eda(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    config = config or load_config()
    bookings, stay_nights, summary = run_preprocessing(config)
    output_dir = _project_path(config["paths"]["eda_results"])
    plots = create_eda_plots(bookings, output_dir, config)
    summary.to_csv(output_dir / "segment_summary.csv", index=False)
    stay_nights.groupby("segment", observed=False).size().rename("expanded_stay_nights").to_csv(
        output_dir / "stay_nights_by_segment.csv"
    )
    return summary, stay_nights, plots


if __name__ == "__main__":
    summary, stay_nights, plots = run_eda()
    print(summary.to_string(index=False))
    print(f"Expanded stay nights: {len(stay_nights):,}")
    print(f"Created {len(plots)} plots in results/eda")
