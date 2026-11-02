"""Hotel star-tier assignment and price/room-size calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import PROJECT_ROOT, load_config
from src.data_prep import run_preprocessing


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def assign_tier_from_stars(stars: pd.Series, config: dict[str, Any]) -> pd.Series:
    """Map Expedia property star ratings to the three configured tiers."""
    assignments = pd.Series(index=stars.index, dtype="object")
    for tier_name, star_values in config["data"]["expedia_tier_star_bins"].items():
        assignments.loc[stars.isin(star_values)] = tier_name
    return assignments


def fit_expedia_calibration(
    expedia: pd.DataFrame, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Fit hotel-level reference-price spreads and booking response by tier."""
    columns = config["data"]["expedia_columns"]
    required = [
        columns["property_id"],
        columns["star_rating"],
        columns["price"],
        columns["booking_label"],
    ]
    missing = sorted(set(required) - set(expedia.columns))
    if missing:
        raise ValueError(f"Expedia data is missing required columns: {missing}")

    data = expedia[required].copy()
    data[columns["price"]] = pd.to_numeric(data[columns["price"]], errors="coerce")
    data["tier"] = assign_tier_from_stars(data[columns["star_rating"]], config)
    data = data.loc[data["tier"].notna() & data[columns["price"]].gt(0)].copy()
    fitted: dict[str, dict[str, Any]] = {}

    for tier_name in config["tiers"]:
        group = data.loc[data["tier"].eq(tier_name)].copy()
        if group.empty:
            fitted[tier_name] = {"status": "no_rows_for_tier"}
            continue
        property_medians = (
            group.groupby(columns["property_id"])[columns["price"]]
            .agg(hotel_reference_price_usd="median", n_observations="size")
            .reset_index()
        )
        tier_reference = float(property_medians["hotel_reference_price_usd"].median())
        pooling_k = config["data"]["expedia_partial_pooling_k"]
        shrunk = (
            property_medians["n_observations"] * property_medians["hotel_reference_price_usd"]
            + pooling_k * tier_reference
        ) / (property_medians["n_observations"] + pooling_k)
        log_spread = np.log(shrunk / tier_reference)
        spread = float(log_spread.std(ddof=1)) if len(log_spread) > 1 else 0.0

        relative_price = (group[columns["price"]] / tier_reference).to_numpy().reshape(-1, 1)
        booking = pd.to_numeric(group[columns["booking_label"]], errors="coerce").fillna(0).astype(int)
        if booking.nunique() == 2:
            model = LogisticRegression(
                random_state=config["project"]["random_seed"],
                max_iter=config["tiering"]["logistic_max_iterations"],
            )
            model.fit(relative_price, booking)
            coefficients = {
                "intercept": float(model.intercept_[0]),
                "relative_price": float(model.coef_[0, 0]),
            }
            logistic_status = "fitted"
        else:
            coefficients = None
            logistic_status = "insufficient_label_variation"

        fitted[tier_name] = {
            "status": "fitted",
            "reference_price_usd": tier_reference,
            "property_count": int(property_medians.shape[0]),
            "observation_count": int(group.shape[0]),
            "partial_pooling_k": float(pooling_k),
            "hotel_log_price_spread": spread,
            "logistic_status": logistic_status,
            "logistic_coefficients": coefficients,
        }
    return fitted


def estimate_size_betas(
    bookings: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Estimate segment size-value elasticities using an explicit room-code proxy.

    The booking data has no room area. The most frequent room codes are selected,
    sorted by median ADR within segment, and paired with the configured room sizes
    from smallest to largest.
    """
    room_column = config["data"]["hbd_room_type_column"]
    mapping_count = config["data"]["hbd_room_type_mapping_count"]
    if config["data"]["hbd_room_type_mapping_selection"] != "most_frequent_overall":
        raise ValueError("Unsupported HBD room-type mapping selection")
    if config["data"]["hbd_room_type_size_order"] != "segment_median_adr_ascending":
        raise ValueError("Unsupported HBD room-type size ordering")
    if config["data"]["hbd_beta_pair_aggregation"] != "median":
        raise ValueError("Unsupported HBD beta aggregation")
    selected_types = (
        bookings[room_column].value_counts().head(mapping_count).index.tolist()
    )
    estimates: dict[str, dict[str, Any]] = {}
    mappings: dict[str, list[str]] = {}

    for tier_name, tier in config["tiers"].items():
        room_sizes = sorted(room["size_m2"] for room in tier["rooms"])
        estimates[tier_name] = {}
        for segment_name in ["leisure", "business"]:
            segment = bookings.loc[
                bookings["segment"].eq(segment_name)
                & bookings[room_column].isin(selected_types)
                & bookings["adr"].gt(0)
            ]
            adr_by_type = segment.groupby(room_column)["adr"].median().dropna()
            ordered_types = adr_by_type.sort_values().index.tolist()
            if len(ordered_types) < mapping_count:
                estimates[tier_name][segment_name] = {
                    "status": "insufficient_room_type_coverage",
                    "beta": float(tier["segments"][segment_name]["size_beta"]),
                    "source": "user_spec_default",
                }
                continue

            ordered_types = ordered_types[:mapping_count]
            adr_values = adr_by_type.loc[ordered_types].to_numpy(dtype=float)
            size_values = np.asarray(room_sizes[: len(ordered_types)], dtype=float)
            ratios = np.log(adr_values[1:] / adr_values[0]) / np.log(
                size_values[1:] / size_values[0]
            )
            beta = float(np.median(ratios))
            mapping = [str(room_type) for room_type in ordered_types]
            mappings[f"{tier_name}:{segment_name}"] = mapping
            estimates[tier_name][segment_name] = {
                "status": "estimated_proxy",
                "beta": beta,
                "source": "hotel_booking_demand_room_type_adr_proxy",
                "room_type_codes_small_to_large": mapping,
                "median_adr_by_code": [float(value) for value in adr_values],
                "assumed_room_sizes_m2_small_to_large": [float(value) for value in size_values],
                "pairwise_beta_values": [float(value) for value in ratios],
            }
    return estimates, mappings


def run_tiering(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Update the profile artifact with available Expedia and HBD estimates."""
    config = config or load_config()
    profile_path = _project_path(config["paths"]["tier_profiles"])
    with profile_path.open("r", encoding="utf-8") as stream:
        profiles = json.load(stream)

    processed_path = _project_path(config["paths"]["hotel_bookings"]).with_name(
        "processed_bookings.csv"
    )
    if processed_path.exists():
        bookings = pd.read_csv(processed_path)
    else:
        bookings, _, _ = run_preprocessing(config)

    beta_estimates, mappings = estimate_size_betas(bookings, config)
    expedia_path = _project_path(config["paths"]["expedia"])
    expedia_status: dict[str, Any]
    if expedia_path.exists():
        columns = list(config["data"]["expedia_columns"].values())
        expedia = pd.read_csv(expedia_path, usecols=columns)
        expedia_status = {
            "source": "expedia_personalized_sort",
            "status": "available",
            "tiers": fit_expedia_calibration(expedia, config),
        }
    else:
        expedia_status = {
            "source": "expedia_personalized_sort",
            "status": "unavailable_no_local_data",
            "reason": "Kaggle competition data requires authenticated rule acceptance; defaults retained.",
            "tiers": {
                tier_name: {
                    "reference_price_source": "user_spec_default_thb",
                    "reference_price_usd": None,
                    "logistic_status": "not_fitted",
                    "hotel_spread_source": "user_spec_default_spreads",
                    "hotel_log_price_spread": None,
                }
                for tier_name in config["tiers"]
            },
        }

    for tier_name, tier in profiles["tiers"].items():
        tier["beta_calibration"] = beta_estimates[tier_name]
        sources = tier.setdefault("sources", {})
        sources.update(
            {
                "reference_price_thb": "user_spec_default",
                "business_share": "user_spec_default",
                "demand_per_room_episode": "user_spec_default",
                "variable_cost_thb": "user_spec_default",
                "ota_share_leisure": "user_spec_default_pending_phase3_hbd_measurement",
                "ota_share_business": "user_spec_default_pending_phase3_hbd_measurement",
                "rooms": "user_spec_default",
            }
        )
        for spread_name in tier["spreads"]:
            sources[f"spread_{spread_name}"] = "user_spec_default_pending_expedia"
        for room in tier["rooms"]:
            room["sources"] = {"size_m2": "user_spec_default", "count": "user_spec_default"}
        for segment_name, estimate in beta_estimates[tier_name].items():
            if estimate["status"] == "estimated_proxy":
                tier["segments"][segment_name]["size_beta"] = estimate["beta"]
            tier["segments"][segment_name]["size_beta_source"] = estimate["source"]
            segment_sources = tier["segments"][segment_name].setdefault("sources", {})
            segment_sources.update(
                {
                    "wtp_median_reference": "user_spec_default_pending_phase3_hbd_fit",
                    "wtp_sigma": "user_spec_default_pending_phase3_hbd_fit",
                    "size_beta": estimate["source"],
                    "cancel_base": "user_spec_default_pending_phase3_hbd_fit",
                }
            )

        expedia_tier = expedia_status["tiers"][tier_name]
        tier["expedia_calibration"] = expedia_tier
        if expedia_status["status"] == "available" and expedia_tier.get("status") == "fitted":
            tier["reference_price_usd"] = expedia_tier["reference_price_usd"]
            tier["spreads"]["reference_price"] = expedia_tier["hotel_log_price_spread"]
            tier["sources"]["reference_price_usd"] = "expedia_hotel_median_price_usd"
            tier["sources"]["reference_price_spread"] = "expedia_partial_pooling_log_spread"
            tier["sources"]["logistic_price_response"] = "expedia_booking_bool_logistic"
        else:
            tier["sources"]["reference_price_usd"] = "expedia_unavailable_user_spec_default"
            tier["sources"]["reference_price_spread"] = "expedia_unavailable_user_spec_default"
            tier["sources"]["logistic_price_response"] = "expedia_unavailable_not_fitted"

    profiles["calibration_status"] = "hbd_beta_calibrated_expedia_defaults"
    profiles["calibration"] = {
        "hotel_booking_demand": {
            "status": "estimated_room_size_beta_proxy",
            "mapping_selection": config["data"]["hbd_room_type_mapping_selection"],
            "size_ordering": config["data"]["hbd_room_type_size_order"],
            "pairwise_beta_aggregation": config["data"]["hbd_beta_pair_aggregation"],
            "mapping_count": config["data"]["hbd_room_type_mapping_count"],
            "mapping_by_tier_and_segment": mappings,
            "data_has_room_area_or_star_tier": False,
        },
        "expedia": expedia_status,
    }
    with profile_path.open("w", encoding="utf-8") as stream:
        json.dump(profiles, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    output_dir = _project_path(config["paths"]["results"]) / "tiering"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tier_name, segments in beta_estimates.items():
        for segment_name, estimate in segments.items():
            rows.append(
                {
                    "tier": tier_name,
                    "segment": segment_name,
                    "size_beta": estimate["beta"],
                    "status": estimate["status"],
                    "room_type_codes": ",".join(estimate.get("room_type_codes_small_to_large", [])),
                    "expedia_status": expedia_status["status"],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "tier_calibration_summary.csv", index=False)
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sim_config.yaml")
    args = parser.parse_args()
    profiles = run_tiering(load_config(args.config))
    print(f"Calibration status: {profiles['calibration_status']}")
    print(pd.read_csv(PROJECT_ROOT / "results" / "tiering" / "tier_calibration_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
