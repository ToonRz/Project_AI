"""Streamlit demonstration for the two-agent hotel pricing lab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.baselines import FixedPolicy, RuleBasedPolicy
from src.config import PROJECT_ROOT, load_config
from src.env import HotelPricingEnv
from src.evaluate import run_stress_tests
from src.hotels import generate_hotel_population
from src.rl_agent import QTablePolicy, TabularQLearner


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def policy_factory(
    name: str, q_table: np.ndarray, q_counts: np.ndarray, config: dict
) -> object:
    if name == "Fixed":
        return FixedPolicy(config)
    if name == "Rule-based":
        return RuleBasedPolicy(config)
    learner = TabularQLearner(
        config,
        algorithm="n_step",
        q_table=q_table,
        update_counts=q_counts,
    )
    return QTablePolicy(learner, config, seed=config["project"]["random_seed"])


def run_interactive_replay(
    tier: str,
    season: str,
    weekend: bool,
    hotel_index: int,
    policy_name: str,
    q_table: np.ndarray,
    q_counts: np.ndarray,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _, test_hotels = generate_hotel_population(config)
    hotels = test_hotels[tier]
    hotel = hotels[hotel_index % len(hotels)]
    tier_id = config["tiers"][tier]["id"]
    seed = (
        config["evaluation"]["seed_base"]
        + config["evaluation"]["seed_episode_multiplier"] * hotel_index
        + tier_id
    )
    env = HotelPricingEnv(hotel=hotel, config=config)
    env.reset(
        seed=seed,
        options={
            "hotel": hotel,
            "season": season,
            "is_weekend": weekend,
            "scenario_seed": seed,
        },
    )
    policy = policy_factory(policy_name, q_table, q_counts, config)
    hourly_rows = []
    customer_rows = []
    step = 0
    while True:
        action = policy.select_actions(env)
        _, _, terminated, truncated, info = env.step(action)
        sale_day = step // config["episode"]["hours_per_day"] + 1
        hour = step % config["episode"]["hours_per_day"]
        for room_index, room in enumerate(hotel.rooms):
            hourly_rows.append(
                {
                    "step": step,
                    "sale_day": sale_day,
                    "hour_of_day": hour,
                    "room_name": room.name,
                    "price_thb": info["prices"][room_index],
                    "rooms_left": info["rooms_left"][room_index],
                    "bookings": sum(
                        booking["room_index"] == room_index for booking in info["bookings"]
                    ),
                }
            )
        for customer in info["customer_log"]:
            customer_rows.append(
                {
                    "step": step,
                    "sale_day": sale_day,
                    "hour_of_day": hour,
                    **customer,
                }
            )
        step += 1
        if terminated or truncated:
            return (
                pd.DataFrame(hourly_rows),
                pd.DataFrame(customer_rows),
                info["episode_metrics"],
            )


st.set_page_config(page_title="ระบบกำหนดราคาโรงแรม", layout="wide")
st.title("ระบบกำหนดราคาโรงแรมแบบ Dynamic Pricing")
st.caption("จำลองลูกค้าและเปรียบเทียบเอเจนต์ reinforcement learning กับ baseline")

config = load_config()
results_dir = project_path(config["paths"]["results"])
model_labels = config["evaluation"]["model_variants"]
model_label = st.sidebar.selectbox("โมเดล RL", list(model_labels))
model_path = project_path(model_labels[model_label])
if not model_path.exists():
    st.error(f"ไม่พบ Q-table: {model_path} กรุณาฝึกโมเดลก่อนเปิด demo")
    st.stop()
q_table = np.load(model_path)
counts_path = model_path.with_name(model_path.name.replace("q_table", "q_counts", 1))
q_counts = np.load(counts_path) if counts_path.exists() else None
if q_counts is None:
    st.error(f"Training update counts not found: {counts_path}")
    st.stop()

tier = st.sidebar.selectbox("ระดับโรงแรม", list(config["tiers"]))
season = st.sidebar.selectbox("ฤดูกาล", config["episode"]["season_names"])
weekend = st.sidebar.checkbox("วันศุกร์หรือวันเสาร์")
_, test_hotels = generate_hotel_population(config)
hotel_index = st.sidebar.number_input(
    "ลำดับโรงแรมทดสอบที่ไม่เคยเห็น",
    min_value=0,
    max_value=len(test_hotels[tier]) - 1,
    value=config["evaluation"]["policy_replay_hotel_index"],
    step=1,
)

st.header("ผลประเมินบนโรงแรมที่ไม่เคยเห็น")
episodes_path = results_dir / "phase6_main_episodes.csv"
if episodes_path.exists():
    episodes = pd.read_csv(episodes_path)
    final_episodes_path = results_dir / "final" / "final_main_episodes.csv"
    if final_episodes_path.exists():
        episodes = pd.concat([episodes, pd.read_csv(final_episodes_path)], ignore_index=True)
    selected_episodes = episodes.loc[
        episodes["model"].eq(model_label)
        & episodes["tier"].eq(tier)
        & episodes["season"].eq(season)
        & episodes["is_weekend"].eq(weekend)
        & episodes["policy"].isin(["Fixed", "Rule-based", "RL"])
    ]
    if not selected_episodes.empty:
        kpi_columns = [
            "revenue",
            "net_profit",
            "occupancy",
            "adr",
            "revpar",
            "profit_per_available_room",
        ]
        selected = selected_episodes.groupby("policy", as_index=False)[kpi_columns].mean()
        counts = selected_episodes.groupby("policy").size().rename("episodes")
        selected["episodes"] = selected["policy"].map(counts)
        fixed_profit = float(selected.loc[selected["policy"].eq("Fixed"), "net_profit"].iloc[0])
        selected["net_profit_pct_vs_fixed"] = (
            selected["net_profit"] / fixed_profit - 1.0
        ) * 100.0
    else:
        selected = pd.DataFrame()
    if not selected.empty:
        display_kpis = selected[
            [
                "policy",
                "revenue",
                "net_profit",
                "occupancy",
                "adr",
                "revpar",
                "profit_per_available_room",
                "net_profit_pct_vs_fixed",
                "episodes",
            ]
        ].rename(
            columns={
                "policy": "นโยบาย",
                "revenue": "รายได้สุทธิหลังคืนเงิน (บาท)",
                "net_profit": "กำไรสุทธิ (บาท)",
                "occupancy": "อัตราเข้าพัก",
                "adr": "ADR",
                "revpar": "RevPAR",
                "profit_per_available_room": "กำไรต่อห้องที่เปิดขาย",
                "net_profit_pct_vs_fixed": "% กำไรเทียบ Fixed",
                "episodes": "จำนวน episode",
            }
        )
        st.caption(f"เงื่อนไข: {season}, {'weekend' if weekend else 'weekday'}")
        st.dataframe(
            display_kpis,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("ไม่มี episode ใน season/weekend ที่เลือก")
else:
    kpi_path = results_dir / "phase6_main_kpis.csv"
    if not kpi_path.exists():
        st.info("ยังไม่พบผลการประเมินหลักใน results/phase6_main_kpis.csv")
    else:
        kpis = pd.read_csv(kpi_path)
        selected = kpis.loc[
            kpis["model"].eq(model_label)
            & kpis["tier"].eq(tier)
            & kpis["policy"].isin(["Fixed", "Rule-based", "RL"])
        ]
        display_kpis = selected[
            [
                "policy",
                "revenue",
                "net_profit",
                "occupancy",
                "adr",
                "revpar",
                "profit_per_available_room",
                "net_profit_pct_vs_fixed",
                "episodes",
            ]
        ].rename(
            columns={
                "policy": "นโยบาย",
                "revenue": "รายได้สุทธิหลังคืนเงิน (บาท)",
                "net_profit": "กำไรสุทธิ (บาท)",
                "occupancy": "อัตราเข้าพัก",
                "adr": "ADR",
                "revpar": "RevPAR",
                "profit_per_available_room": "กำไรต่อห้องที่เปิดขาย",
                "net_profit_pct_vs_fixed": "% กำไรเทียบ Fixed",
                "episodes": "จำนวน episode",
            }
        )
        st.dataframe(
            display_kpis,
            use_container_width=True,
            hide_index=True,
        )

st.header("แผนที่ราคาแนะนำของ policy")
heatmap_path = results_dir / f"phase6_{model_label}_policy_heatmap.csv"
if heatmap_path.exists():
    heatmap = pd.read_csv(heatmap_path)
    weekend_filter = heatmap["is_weekend"].eq(weekend)
    selected_heatmap = heatmap.loc[
        heatmap["tier"].eq(tier)
        & heatmap["season"].eq(season)
        & weekend_filter
    ]
    if not selected_heatmap.empty:
        rooms = list(config["tiers"][tier]["rooms"])
        columns = st.columns(len(rooms))
        for column, room in zip(columns, rooms):
            room_data = selected_heatmap.loc[
                selected_heatmap["room_name"].eq(room["name"])
            ]
            table = room_data.pivot_table(
                index="days_remaining_representative",
                columns="period",
                values="markup",
                aggfunc="mean",
            )
            with column:
                st.markdown(f"**{room['name']} · ตัวคูณราคาอ้างอิง**")
                st.dataframe(table.round(2), use_container_width=True)
else:
    st.info("ยังไม่พบ heatmap สำหรับโมเดลนี้ กรุณารัน evaluation พร้อม --heatmap")

st.header("เล่นซ้ำ episode รายชั่วโมง")
replay_policy = st.selectbox("Policy ที่ใช้เล่นซ้ำ", ["Fixed", "Rule-based", "RL"])
if st.button("เริ่มจำลอง 30 วัน"):
    with st.spinner("กำลังจำลอง 720 ชั่วโมงด้วย customer scenario ที่กำหนด seed แล้ว"):
        hourly, customer_log, metrics = run_interactive_replay(
            tier,
            season,
            weekend,
            int(hotel_index),
            replay_policy,
            q_table,
            q_counts,
            config,
        )
    st.session_state["replay_hourly"] = hourly
    st.session_state["replay_customers"] = customer_log
    st.session_state["replay_metrics"] = metrics

if "replay_hourly" in st.session_state:
    hourly = st.session_state["replay_hourly"]
    customer_log = st.session_state["replay_customers"]
    metrics = st.session_state["replay_metrics"]
    st.metric("กำไรสุทธิ (บาท)", f"{metrics['net_profit']:,.0f}")
    step_index = st.slider(
        "ชั่วโมงใน episode",
        min_value=0,
        max_value=config["episode"]["steps"] - 1,
        value=0,
    )
    hour_view = hourly.loc[hourly["step"].eq(step_index)]
    st.dataframe(hour_view, use_container_width=True, hide_index=True)
    customer_view = customer_log.loc[customer_log["step"].eq(step_index)]
    st.markdown("**การตัดสินใจของลูกค้าในชั่วโมงนี้**")
    st.dataframe(customer_view, use_container_width=True, hide_index=True)
    chart = hourly.pivot(index="step", columns="room_name", values="price_thb")
    st.line_chart(chart)

st.header("ทดสอบ stress scenarios")
if st.button("รัน stress test ทั้ง 5 สถานการณ์"):
    with st.spinner("กำลังรัน stress test บน test hotels ทุก tier"):
        _, stress_kpis = run_stress_tests(q_table, config, q_counts)
    st.dataframe(
        stress_kpis.loc[
            stress_kpis["tier"].eq(tier),
            ["world", "policy", "net_profit", "net_profit_pct_vs_fixed", "episodes"],
        ],
        use_container_width=True,
        hide_index=True,
    )
elif (results_dir / "phase6_stress_kpis.csv").exists():
    stress_kpis = pd.read_csv(results_dir / "phase6_stress_kpis.csv")
    st.dataframe(
        stress_kpis.loc[
            stress_kpis["tier"].eq(tier),
            ["world", "policy", "net_profit", "net_profit_pct_vs_fixed", "episodes"],
        ],
        use_container_width=True,
        hide_index=True,
    )
