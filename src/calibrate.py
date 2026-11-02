"""Fit customer-arrival, WTP, cancellation, channel, and season profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from src.config import PROJECT_ROOT, load_config
from src.data_prep import run_preprocessing


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _arrival_days(bookings: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    lead = pd.to_numeric(bookings["lead_time"], errors="coerce")
    return lead.clip(
        lower=config["calibration"]["arrival_lead_days_min"],
    ).round().astype("Int64").clip(lower=1)


def _default_arrival_weights(segment: str, config: dict[str, Any]) -> np.ndarray:
    horizon = config["episode"]["booking_horizon_days"]
    segment_config = config["tiers"]["budget"]["segments"][segment]["hour_profile"]
    days = np.arange(1, horizon + 1, dtype=float)
    if segment == "leisure":
        center = segment_config["center"]
        spread = segment_config["center_sigma"]
        weights = np.exp(-((days - center) ** 2) / (2 * spread**2))
    else:
        decay = segment_config["arrival_decay_days"]
        weights = np.exp(-((days - 1) / decay))
    return weights / weights.sum()


def fit_arrival_weights(
    bookings: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """Estimate normalized 30-day lead-time weights from observed bookings."""
    horizon = config["episode"]["booking_horizon_days"]
    conversion_rate = config["data"]["conversion_rate"]
    arrival_day = _arrival_days(bookings, config)
    weights: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for segment in config["tiers"]["budget"]["segments"]:
        subset = bookings.loc[bookings["segment"].eq(segment)].copy()
        days = arrival_day.loc[subset.index]
        in_window = days.between(1, horizon)
        counts = days.loc[in_window].value_counts().reindex(range(1, horizon + 1), fill_value=0)
        if counts.sum() == 0:
            distribution = _default_arrival_weights(segment, config)
            source = "user_spec_default_shape"
        else:
            distribution = counts.to_numpy(dtype=float) / counts.sum()
            source = "hotel_booking_demand_lead_time"
        weights[segment] = distribution.tolist()
        for day, weight in enumerate(distribution, start=1):
            booking_count = int(counts.loc[day]) if counts.sum() > 0 else 0
            rows.append(
                {
                    "segment": segment,
                    "days_before_arrival": day,
                    "booking_count": booking_count,
                    "estimated_visitors": booking_count / conversion_rate,
                    "arrival_weight": float(weight),
                    "source": source,
                }
            )
    return weights, pd.DataFrame(rows)


def _fit_cancellation_segment(
    bookings: pd.DataFrame, segment: str, config: dict[str, Any]
) -> dict[str, Any]:
    feature = config["calibration"]["cancellation_deposit_feature"]
    max_lead = config["calibration"]["cancellation_fit_lead_days_max"]
    data = bookings.loc[
        bookings["segment"].eq(segment) & bookings["lead_time"].between(0, max_lead)
    ].copy()
    data[feature] = data[feature].fillna("Unknown").astype(str)
    deposit_types = sorted(data[feature].unique().tolist())
    design = pd.DataFrame(index=data.index)
    design["lead_time_scaled"] = data["lead_time"].astype(float) / max_lead
    for deposit_type in deposit_types:
        design[f"deposit_type={deposit_type}"] = data[feature].eq(deposit_type).astype(float)
    labels = data["is_canceled"].astype(int)
    if labels.nunique() != 2 or data.empty:
        return {
            "status": "insufficient_label_variation",
            "cancel_rate": float(labels.mean()) if len(labels) else 0.0,
            "cancel_base": float(labels.mean()) if len(labels) else 0.0,
            "deposit_shares": data[feature].value_counts(normalize=True).to_dict(),
        }

    model = LogisticRegression(
        random_state=config["project"]["random_seed"],
        max_iter=config["calibration"]["cancellation_logistic_max_iterations"],
    )
    model.fit(design, labels)
    fitted_probabilities = model.predict_proba(design)[:, 1]
    deposit_shares = data[feature].value_counts(normalize=True).to_dict()

    horizon = config["episode"]["booking_horizon_days"]
    lead_factor_start = config["demand"]["cancellation_lead_factor_start"]
    lead_factor_slope = config["demand"]["cancellation_lead_factor_slope"]
    lead_factors = []
    predicted_cancellation = []
    for days in range(1, horizon + 1):
        factor = lead_factor_start + lead_factor_slope * days / horizon
        lead_factors.append(factor)
        marginal = 0.0
        for deposit_type, share in deposit_shares.items():
            row = {"lead_time_scaled": days / max_lead}
            row.update(
                {
                    f"deposit_type={name}": float(name == deposit_type)
                    for name in deposit_types
                }
            )
            row_frame = pd.DataFrame([row], columns=design.columns)
            marginal += float(share) * float(model.predict_proba(row_frame)[0, 1])
        predicted_cancellation.append(marginal)
    factors = np.asarray(lead_factors, dtype=float)
    predicted = np.asarray(predicted_cancellation, dtype=float)
    fitted_base = float(np.dot(factors, predicted) / np.dot(factors, factors))
    clip_min, clip_max = config["calibration"]["cancellation_probability_clip"]
    fitted_base = float(np.clip(fitted_base, clip_min, clip_max))

    coefficient = {name: float(value) for name, value in zip(design.columns, model.coef_[0])}
    return {
        "status": "fitted",
        "observation_count": int(len(data)),
        "cancel_rate": float(labels.mean()),
        "cancel_base": fitted_base,
        "deposit_shares": {str(key): float(value) for key, value in deposit_shares.items()},
        "logistic": {
            "intercept": float(model.intercept_[0]),
            "coefficients": coefficient,
            "feature_names": design.columns.tolist(),
            "roc_auc_in_sample": float(roc_auc_score(labels, fitted_probabilities)),
            "log_loss_in_sample": float(log_loss(labels, fitted_probabilities)),
        },
        "predicted_cancel_rate_by_days_remaining": predicted.tolist(),
    }


def fit_cancellation_profiles(
    bookings: pd.DataFrame, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    segments = config["tiers"]["budget"]["segments"]
    return {
        segment: _fit_cancellation_segment(bookings, segment, config)
        for segment in segments
    }


def fit_season_and_weekday_factors(
    bookings: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Estimate relative arrival-date booking rates by configured season and weekend."""
    data = bookings.loc[bookings["arrival_date"].notna()].copy()
    season_months = config["episode"]["season_months"]
    month_to_season = {
        month: season for season, months in season_months.items() for month in months
    }
    data["season"] = data["arrival_date"].dt.month.map(month_to_season)
    data["is_weekend"] = data["arrival_weekday"].isin(config["episode"]["weekend_weekdays"])

    hotel_names = data["hotel"].dropna().unique().tolist()
    dates = pd.date_range(data["arrival_date"].min(), data["arrival_date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product([hotel_names, dates], names=["hotel", "arrival_date"])
    observed = (
        data.groupby(["hotel", "arrival_date", "segment"], observed=False)
        .size()
        .unstack("segment", fill_value=0)
    )
    daily = observed.reindex(full_index, fill_value=0).reset_index()
    daily["season"] = daily["arrival_date"].dt.month.map(month_to_season)
    daily["is_weekend"] = daily["arrival_date"].dt.day_name().isin(
        config["episode"]["weekend_weekdays"]
    )

    season_factors: dict[str, dict[str, float]] = {}
    weekend_factors: dict[str, dict[str, float]] = {"weekday": {}, "weekend": {}}
    for segment in config["tiers"]["budget"]["segments"]:
        daily[segment] = daily.get(segment, 0)
        overall_rate = float(daily[segment].mean())
        seasonal = daily.groupby("season", observed=False)[segment].mean() / overall_rate
        season_factors[segment] = {
            name: float(seasonal.get(name, 1.0)) for name in config["episode"]["season_names"]
        }
        for label, is_weekend in [("weekday", False), ("weekend", True)]:
            value = daily.loc[daily["is_weekend"].eq(is_weekend), segment].mean()
            weekend_factors[label][segment] = float(value / overall_rate)
    return season_factors, weekend_factors


def validate_leave_one_hotel_out(
    bookings: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Compare lead-time curves trained on one HBD hotel with the other hotel."""
    horizon = config["episode"]["booking_horizon_days"]
    all_hotels = sorted(bookings["hotel"].dropna().unique().tolist())
    rows = []
    for held_out in all_hotels:
        train = bookings.loc[bookings["hotel"].ne(held_out)]
        test = bookings.loc[bookings["hotel"].eq(held_out)]
        for segment in config["tiers"]["budget"]["segments"]:
            train_days = _arrival_days(train.loc[train["segment"].eq(segment)], config)
            test_days = _arrival_days(test.loc[test["segment"].eq(segment)], config)
            train_days = train_days.loc[train_days.between(1, horizon)]
            test_days = test_days.loc[test_days.between(1, horizon)]
            train_curve = train_days.value_counts().reindex(range(1, horizon + 1), fill_value=0).to_numpy(dtype=float)
            test_curve = test_days.value_counts().reindex(range(1, horizon + 1), fill_value=0).to_numpy(dtype=float)
            train_curve /= max(train_curve.sum(), 1.0)
            test_curve /= max(test_curve.sum(), 1.0)
            cumulative_gap = np.abs(np.cumsum(train_curve) - np.cumsum(test_curve))
            rows.append(
                {
                    "held_out_hotel": held_out,
                    "segment": segment,
                    "train_booking_count": int(len(train_days)),
                    "test_booking_count": int(len(test_days)),
                    "mean_absolute_curve_error": float(np.mean(np.abs(train_curve - test_curve))),
                    "max_cumulative_share_error": float(cumulative_gap.max()),
                    "ks_statistic": float(cumulative_gap.max()),
                }
            )
    return pd.DataFrame(rows)


def _fit_wtp(bookings: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    estimates = {}
    for segment in config["tiers"]["budget"]["segments"]:
        adr = bookings.loc[bookings["segment"].eq(segment) & bookings["adr"].gt(0), "adr"].astype(float)
        log_adr = np.log(adr)
        estimates[segment] = {
            "mu_log_adr": float(log_adr.mean()),
            "sigma_log_adr": float(log_adr.std(ddof=config["calibration"]["wtp_log_std_ddof"])),
            "geometric_mean_adr_source_currency": float(np.exp(log_adr.mean())),
            "observation_count": int(log_adr.size),
        }
    return estimates


def run_calibration(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit available customer parameters and write the tier-profile artifact."""
    config = config or load_config()
    processed_path = _project_path(config["paths"]["hotel_bookings"]).with_name(
        "processed_bookings.csv"
    )
    if processed_path.exists():
        bookings = pd.read_csv(processed_path, parse_dates=["arrival_date"])
    else:
        bookings, _, _ = run_preprocessing(config)

    profiles_path = _project_path(config["paths"]["tier_profiles"])
    with profiles_path.open("r", encoding="utf-8") as stream:
        profiles = json.load(stream)

    arrival_weights, arrival_table = fit_arrival_weights(bookings, config)
    wtp = _fit_wtp(bookings, config)
    cancellation = fit_cancellation_profiles(bookings, config)
    season_factors, dow_factors = fit_season_and_weekday_factors(bookings, config)
    loo = validate_leave_one_hotel_out(bookings, config)
    segment_summary = (
        bookings.groupby("segment", observed=False)
        .agg(ota_share=("is_ota", "mean"), booking_share=("segment", "size"))
    )
    segment_summary["booking_share"] /= segment_summary["booking_share"].sum()

    default_cancel_mean = {
        segment: float(
            np.mean([tier["segments"][segment]["cancel_base"] for tier in config["tiers"].values()])
        )
        for segment in cancellation
    }
    for tier_name, tier_profile in profiles["tiers"].items():
        tier_profile["arrival_weights"] = arrival_weights
        tier_profile["season_multipliers"] = season_factors
        tier_profile["dow_multipliers"] = dow_factors
        tier_profile["sources"]["arrival_weights"] = "hotel_booking_demand_lead_time"
        tier_profile["sources"]["season_multipliers"] = "hotel_booking_demand_arrival_dates"
        tier_profile["sources"]["dow_multipliers"] = "hotel_booking_demand_arrival_dates"
        tier_profile["sources"]["business_share"] = "user_spec_default_no_star_tier_mapping"

        for segment, segment_profile in tier_profile["segments"].items():
            segment_profile["wtp_calibration"] = wtp[segment]
            segment_profile["wtp_sigma"] = wtp[segment]["sigma_log_adr"]
            segment_profile["sources"]["wtp_sigma"] = "hotel_booking_demand_log_adr_std"
            segment_profile["sources"]["wtp_median_reference"] = (
                "user_spec_default_no_star_tier_currency_mapping"
            )
            ota_share = float(segment_summary.loc[segment, "ota_share"])
            tier_profile[f"ota_share_{segment}"] = ota_share
            tier_profile["sources"][f"ota_share_{segment}"] = "hotel_booking_demand_distribution_channel"

            fitted_base = cancellation[segment]["cancel_base"]
            scale = fitted_base / max(default_cancel_mean[segment], np.finfo(float).eps)
            clip_min, clip_max = config["calibration"]["cancellation_probability_clip"]
            scaled_base = float(
                np.clip(
                    config["tiers"][tier_name]["segments"][segment]["cancel_base"] * scale,
                    clip_min,
                    clip_max,
                )
            )
            segment_profile["cancel_base"] = scaled_base
            segment_profile["cancellation_calibration"] = {
                "pooled_hbd_model": cancellation[segment],
                "tier_default_scale": float(scale),
                "source": "hotel_booking_demand_logistic_scaled_to_tier_default",
            }
            segment_profile["sources"]["cancel_base"] = (
                "hotel_booking_demand_logistic_scaled_to_tier_default"
            )

    profiles["calibration_status"] = "hbd_calibrated_expedia_defaults"
    calibration_record = profiles.setdefault("calibration", {})
    calibration_record["customer"] = {
        "arrival_weights_by_segment": arrival_weights,
        "wtp_lognormal_by_segment": wtp,
        "cancellation_logistic_by_segment": cancellation,
        "ota_share_by_segment": {
            segment: float(segment_summary.loc[segment, "ota_share"]) for segment in segment_summary.index
        },
        "booking_share_by_segment": {
            segment: float(segment_summary.loc[segment, "booking_share"]) for segment in segment_summary.index
        },
        "season_multipliers_by_segment": season_factors,
        "dow_multipliers_by_segment": dow_factors,
        "conversion_rate_used_for_de_censoring": config["data"]["conversion_rate"],
        "conversion_rate_sensitivity": config["data"]["conversion_rate_sensitivity"],
        "loco_metric_note": "Normalized 30-day lead-time curve; booking totals are de-censored by the configured conversion rate.",
    }
    profiles_path.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output_dir = _project_path(config["calibration"]["results_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    arrival_table.to_csv(output_dir / "arrival_curve.csv", index=False)
    loo.to_csv(output_dir / "leave_one_hotel_out.csv", index=False)
    pd.DataFrame(
        [
            {"segment": segment, **values}
            for segment, values in wtp.items()
        ]
    ).to_csv(output_dir / "wtp_lognormal.csv", index=False)
    pd.DataFrame(
        [
            {
                "segment": segment,
                "ota_share": float(segment_summary.loc[segment, "ota_share"]),
                "booking_share": float(segment_summary.loc[segment, "booking_share"]),
                "cancel_rate_within_30_days": cancellation[segment]["cancel_rate"],
                "fitted_cancel_base": cancellation[segment]["cancel_base"],
            }
            for segment in cancellation
        ]
    ).to_csv(output_dir / "customer_calibration_summary.csv", index=False)
    return {
        "profiles": profiles,
        "arrival_curve": arrival_table,
        "leave_one_hotel_out": loo,
        "wtp": wtp,
        "cancellation": cancellation,
        "season_factors": season_factors,
        "dow_factors": dow_factors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sim_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_calibration(config)
    print(f"Calibration status: {result['profiles']['calibration_status']}")
    output_dir = _project_path(config["calibration"]["results_directory"])
    print(pd.read_csv(output_dir / "customer_calibration_summary.csv").to_string(index=False))
    print("Leave-one-hotel-out:")
    print(result["leave_one_hotel_out"].to_string(index=False))


if __name__ == "__main__":
    main()
