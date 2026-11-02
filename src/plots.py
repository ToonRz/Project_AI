"""Create evaluation, learning-curve, replay, and policy heatmap figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config


def _path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _style(config: dict[str, Any]) -> tuple[tuple[float, float], int]:
    settings = config["plots"]
    return tuple(settings["figure_size_inches"]), int(settings["figure_dpi"])


def plot_learning_curves(config: dict[str, Any] | None = None) -> list[Path]:
    config = config or load_config()
    results = _path(config, config["paths"]["results"])
    size, dpi = _style(config)
    inputs = {
        "n_step_domain_randomization": "learning_curve_n_step_domain_randomization.csv",
        "n_step_single": "learning_curve_n_step_single.csv",
        "monte_carlo_domain_randomization": "learning_curve_monte_carlo_domain_randomization.csv",
        "monte_carlo_single": "learning_curve_monte_carlo_single.csv",
    }
    available = {
        label: pd.read_csv(results / filename)
        for label, filename in inputs.items()
        if (results / filename).exists()
    }
    if not available:
        return []
    output = results / "plots"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for tier in config["tiers"]:
        fig, axis = plt.subplots(figsize=size, dpi=dpi)
        for label, table in available.items():
            tier_data = table.loc[table["tier"].eq(tier)]
            if not tier_data.empty:
                axis.plot(
                    tier_data["episode"],
                    tier_data["moving_average_normalized_profit"],
                    label=label.replace("_", " "),
                    linewidth=1.2,
                )
        axis.set(title=f"{tier.title()} learning curve", xlabel="Episode", ylabel="Moving average normalized profit")
        axis.legend(fontsize="small")
        axis.grid(alpha=0.25)
        path = output / f"learning_curve_{tier}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_main_comparison(
    kpis: pd.DataFrame | None = None, config: dict[str, Any] | None = None
) -> Path | None:
    config = config or load_config()
    results = _path(config, config["paths"]["results"])
    source = results / "phase6_main_kpis.csv"
    if kpis is None:
        if not source.exists():
            return None
        kpis = pd.read_csv(source)
    if kpis.empty:
        return None
    _, dpi = _style(config)
    size = tuple(config["plots"]["main_figure_size_inches"])
    fig, axes = plt.subplots(1, len(config["tiers"]), figsize=size, dpi=dpi, squeeze=False)
    model_order = [
        "n_step_domain_randomization",
        "n_step_single",
        "monte_carlo_domain_randomization",
        "monte_carlo_single",
    ]
    model_labels = {
        "n_step_domain_randomization": "n-step DR",
        "n_step_single": "n-step single",
        "monte_carlo_domain_randomization": "MC DR",
        "monte_carlo_single": "MC single",
    }
    for axis, tier in zip(axes[0], config["tiers"]):
        subset = kpis.loc[kpis["tier"].eq(tier)]
        subset = subset.assign(model_label=subset["model"].map(model_labels))
        pivot = subset.pivot(index="model_label", columns="policy", values="net_profit_pct_vs_fixed")
        ordered_labels = [model_labels[model] for model in model_order if model in set(subset["model"])]
        pivot = pivot.reindex(index=ordered_labels, columns=["Fixed", "Rule-based", "RL"])
        pivot.plot.bar(ax=axis, rot=20, width=0.82)
        axis.set_title(tier.title())
        axis.set_ylabel("Net profit vs Fixed (%)")
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelsize=8)
        for tick in axis.get_xticklabels():
            tick.set_ha("right")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize="x-small")
    path = results / "phase6_main_relative_profit.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_stress(kpis: pd.DataFrame | None = None, config: dict[str, Any] | None = None) -> Path | None:
    config = config or load_config()
    results = _path(config, config["paths"]["results"])
    source = results / "phase6_stress_kpis.csv"
    if kpis is None:
        if not source.exists():
            return None
        kpis = pd.read_csv(source)
    size, dpi = _style(config)
    fig, axes = plt.subplots(1, len(config["tiers"]), figsize=size, dpi=dpi, squeeze=False)
    world_order = list(config["evaluation"]["stress_worlds"])
    for axis, tier in zip(axes[0], config["tiers"]):
        subset = kpis.loc[kpis["tier"].eq(tier) & ~kpis["policy"].eq("Fixed")]
        for policy in ["Rule-based", "RL"]:
            line = subset.loc[subset["policy"].eq(policy)].set_index("world").reindex(world_order)
            axis.plot(world_order, line["net_profit_pct_vs_fixed"], marker="o", label=policy)
        axis.set_title(tier.title())
        axis.set_ylabel("Net profit vs Fixed (%)")
        axis.tick_params(axis="x", rotation=45)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.grid(alpha=0.25)
        axis.legend()
    path = results / "phase6_stress_relative_profit.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sensitivity(
    kpis: pd.DataFrame | None = None, config: dict[str, Any] | None = None
) -> Path | None:
    config = config or load_config()
    results = _path(config, config["paths"]["results"])
    source = results / "phase6_sensitivity_kpis.csv"
    if kpis is None:
        if not source.exists():
            return None
        kpis = pd.read_csv(source)
    size, dpi = _style(config)
    parameters = list(kpis["parameter"].drop_duplicates())
    fig, axes = plt.subplots(len(parameters), len(config["tiers"]), figsize=size, dpi=dpi, squeeze=False)
    for row_index, parameter in enumerate(parameters):
        for column_index, tier in enumerate(config["tiers"]):
            axis = axes[row_index, column_index]
            subset = kpis.loc[kpis["parameter"].eq(parameter) & kpis["tier"].eq(tier)]
            for policy in ["Rule-based", "RL"]:
                line = subset.loc[subset["policy"].eq(policy)].sort_values("value")
                axis.plot(line["value"], line["net_profit_pct_vs_fixed"], marker="o", label=policy)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_title(f"{tier.title()} · {parameter}")
            axis.set_xlabel("Scenario value")
            axis.set_ylabel("Net profit vs Fixed (%)")
            axis.grid(alpha=0.25)
            axis.legend()
    path = results / "phase6_sensitivity_relative_profit.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_policy_heatmaps(
    heatmap: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    model_label: str = "n_step_domain_randomization",
) -> list[Path]:
    config = config or load_config()
    results = _path(config, config["paths"]["results"])
    source = results / f"phase6_{model_label}_policy_heatmap.csv"
    if heatmap is None:
        if not source.exists():
            return []
        heatmap = pd.read_csv(source)
    output = results / "policy_heatmaps" / model_label
    output.mkdir(parents=True, exist_ok=True)
    _, dpi = _style(config)
    size = tuple(config["plots"]["heatmap_figure_size_inches"])
    paths: list[Path] = []
    for (tier, season, weekend), subset in heatmap.groupby(["tier", "season", "is_weekend"]):
        fig, axes = plt.subplots(len(config["tiers"][tier]["rooms"]), 1, figsize=size, dpi=dpi, squeeze=False)
        for room_index, axis in enumerate(axes[:, 0]):
            room_data = subset.loc[subset["room_index"].eq(room_index)]
            matrix = room_data.pivot(
                index="day_bucket", columns="period", values="markup"
            ).reindex(index=range(len(config["rl"]["state"]["day_bucket_ranges"])))
            image = axis.imshow(
                matrix.to_numpy(),
                aspect="auto",
                vmin=min(config["pricing"]["markup_levels"]),
                vmax=max(config["pricing"]["markup_levels"]),
                cmap="viridis",
            )
            axis.set_yticks(range(len(matrix.index)))
            axis.set_yticklabels(
                [
                    f"D-{low}" if low == high else f"D-{low}–{high}"
                    for low, high in config["rl"]["state"]["day_bucket_ranges"]
                ]
            )
            axis.set_xticks(range(len(matrix.columns)))
            period_hours = config["rl"]["state"]["time_period_hours"]
            axis.set_xticklabels(
                [
                    f"{period * period_hours:02d}–{(period + 1) * period_hours:02d}"
                    for period in matrix.columns
                ]
            )
            axis.set_ylabel(config["tiers"][tier]["rooms"][room_index]["name"])
            if room_index == len(axes[:, 0]) - 1:
                axis.set_xlabel("Hour of day")
        fig.suptitle(f"{tier.title()} · {season.title()} · {'Weekend' if weekend else 'Weekday'} · greedy markup")
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Markup vs room reference", fraction=0.03, pad=0.02)
        safe_name = f"{tier}_{season}_{'weekend' if weekend else 'weekday'}.png"
        path = output / safe_name
        fig.subplots_adjust(left=0.12, right=0.86, top=0.92, bottom=0.08, hspace=0.38)
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_replay(config: dict[str, Any] | None = None) -> list[Path]:
    config = config or load_config()
    results = _path(config, config["paths"]["results"])
    hourly_path = results / "phase6_replay_hourly.csv"
    daily_path = results / "phase6_replay_daily.csv"
    if not hourly_path.exists() or not daily_path.exists():
        return []
    hourly = pd.read_csv(hourly_path)
    daily = pd.read_csv(daily_path)
    size, dpi = _style(config)
    paths = []
    rl = hourly.loc[hourly["policy"].eq("RL")]
    for tier, subset in rl.groupby("tier"):
        fig, axis = plt.subplots(figsize=size, dpi=dpi)
        for room_name, room_data in subset.groupby("room_name"):
            axis.plot(room_data["step"], room_data["price_thb"], label=room_name)
        axis.set(title=f"{tier.title()} RL hourly prices", xlabel="Hourly step", ylabel="Price (THB)")
        axis.legend()
        axis.grid(alpha=0.25)
        path = results / f"phase6_replay_prices_{tier}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

        daily_rl = daily.loc[(daily["tier"].eq(tier)) & daily["policy"].eq("RL")]
        pivot = daily_rl.pivot_table(index="sale_day", columns="room_name", values="bookings", aggfunc="sum", fill_value=0)
        fig, axis = plt.subplots(figsize=size, dpi=dpi)
        pivot.plot(kind="bar", stacked=True, ax=axis)
        axis.set(title=f"{tier.title()} RL daily bookings", xlabel="Sale day", ylabel="New bookings")
        axis.legend()
        path = results / f"phase6_replay_bookings_{tier}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def create_phase6_figures(config: dict[str, Any] | None = None) -> list[Path]:
    config = config or load_config()
    outputs = plot_learning_curves(config)
    for result in [
        plot_main_comparison(config=config),
        plot_stress(config=config),
        plot_sensitivity(config=config),
    ]:
        if result is not None:
            outputs.append(result)
    for model_label in config["evaluation"]["model_variants"]:
        outputs.extend(plot_policy_heatmaps(config=config, model_label=model_label))
    outputs.extend(plot_replay(config))
    return outputs


if __name__ == "__main__":
    paths = create_phase6_figures()
    print(f"Created {len(paths)} figures")
