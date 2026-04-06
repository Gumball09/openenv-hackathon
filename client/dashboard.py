"""
Streamlit dashboard for the Logistics Crisis Manager environment.

Run with::

    pip install streamlit
    cd /home/shubh/openenv-hackathon
    streamlit run logistics_crisis_manager/client/dashboard.py

Features:
  • Pick a tiered task (easy / medium / hard) and reset
  • Send any of the 3 actions (move_cargo / redeploy_stock / wait)
  • Live news feed, inventory table, in-transit shipments
  • Live grader sub-scores + verification score
  • Optional "Autoplay with LLM" mode that drives the env via the same
    OpenAI/HF Router client used by the baseline script
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Allow `streamlit run logistics_crisis_manager/client/dashboard.py` from
# the parent directory by ensuring the package root is on sys.path.
_THIS = os.path.abspath(__file__)
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from logistics_crisis_manager.models import (  # noqa: E402
    CarrierType,
    LogisticsCrisisManagerObservation,
    MoveCargo,
    RedeployStock,
    Wait,
)
from logistics_crisis_manager.server.environment import (  # noqa: E402
    TASK_CONFIGS,
    LogisticsEnv,
)

# ── Page config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Logistics Crisis Manager",
    page_icon="🚢",
    layout="wide",
)

# ── Session state bootstrap ─────────────────────────────────────────


def _init_state() -> None:
    if "env" not in st.session_state:
        st.session_state.env = LogisticsEnv()
        st.session_state.task_id = "medium"
        st.session_state.last_obs = st.session_state.env.reset(
            task_id=st.session_state.task_id, seed=0
        )
        st.session_state.action_history: List[Dict[str, Any]] = []


_init_state()


# ── Helpers ─────────────────────────────────────────────────────────


def _reset_env(task_id: str, seed: int) -> None:
    st.session_state.env = LogisticsEnv()
    st.session_state.task_id = task_id
    st.session_state.last_obs = st.session_state.env.reset(
        task_id=task_id, seed=seed
    )
    st.session_state.action_history = []


def _apply_action(action: Any) -> None:
    env: LogisticsEnv = st.session_state.env
    try:
        obs = env.step(action)
        st.session_state.last_obs = obs
        st.session_state.action_history.append(
            {
                "step": env.state.step_count,
                "type": action.type,
                "details": action.model_dump_json(exclude={"metadata"}),
                "reward": obs.reward,
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.error(f"Action failed: {exc}")
        st.code(traceback.format_exc())


def _inventory_df(env: LogisticsEnv) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for city, skus in env.state.inventory.items():
        node = env.state.nodes.get(city)
        rows.append(
            {
                "city": city,
                "role": getattr(node, "role", "?"),
                "total_units": sum(skus.values()),
                **skus,
            }
        )
    return pd.DataFrame(rows).fillna(0)


def _shipments_df(env: LogisticsEnv) -> pd.DataFrame:
    rows = []
    for sh in env._shipments:  # noqa: SLF001 — internal but stable
        rows.append(
            {
                "shipment_id": sh.shipment_id,
                "origin": sh.origin,
                "destination": sh.destination,
                "carrier": sh.carrier_type.value,
                "units": sum(sh.skus.values()),
                "depart": sh.depart_time,
                "arrive": sh.arrive_time,
                "ETA (h)": sh.arrive_time - env.state.current_time,
            }
        )
    return pd.DataFrame(rows)


# ── Sidebar — task selection & reset ────────────────────────────────

st.sidebar.title("🚢 Logistics CM")
st.sidebar.caption("OpenEnv Hackathon environment")

with st.sidebar.form("reset_form"):
    new_task = st.selectbox(
        "Task",
        list(TASK_CONFIGS.keys()),
        index=list(TASK_CONFIGS.keys()).index(st.session_state.task_id),
        help="easy = no events · medium = port strike · hard = port + fuel + viral",
    )
    new_seed = st.number_input("Seed", value=0, step=1)
    if st.form_submit_button("🔄 Reset env", use_container_width=True):
        _reset_env(new_task, int(new_seed))
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Task:** `{st.session_state.task_id}`")
st.sidebar.caption(TASK_CONFIGS[st.session_state.task_id]["description"])

env: LogisticsEnv = st.session_state.env
obs: LogisticsCrisisManagerObservation = st.session_state.last_obs
report = env.grader.report()

# ── Top metrics ─────────────────────────────────────────────────────

st.title("Logistics Crisis Manager — Live Dashboard")

col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Step", env.state.step_count)
col_b.metric("Sim hour", env.state.current_time)
col_c.metric("Budget", f"${env.state.budget:,.0f}")
col_d.metric(
    "Deliveries",
    f"{env._deliveries_on_time}/{env._deliveries_total}",
    help="on-time / total",
)
col_e.metric("Verification Score", f"{report['verification_score']:.4f}")

# Sub-score progress bars
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.progress(report["safety"], text=f"safety {report['safety']:.2f}")
sc2.progress(
    report["reliability"], text=f"reliability {report['reliability']:.2f}"
)
sc3.progress(report["efficiency"], text=f"efficiency {report['efficiency']:.2f}")
sc4.progress(
    report["task_progress"], text=f"task progress {report['task_progress']:.2f}"
)

if obs.done:
    st.warning("⛔ Episode finished. Reset to start a new one.")

# ── Two-column body: state | actions ────────────────────────────────

left, right = st.columns([3, 2], gap="large")

# ── Left column: state ──────────────────────────────────────────────

with left:
    st.subheader("📦 Inventory")
    st.dataframe(
        _inventory_df(env), use_container_width=True, hide_index=True
    )

    st.subheader("✈️ In Transit")
    ship_df = _shipments_df(env)
    if ship_df.empty:
        st.caption("_No shipments in transit._")
    else:
        st.dataframe(ship_df, use_container_width=True, hide_index=True)

    st.subheader("📰 News Feed")
    feed = env.state.news_feed[-15:]
    for line in reversed(feed):
        if "ALERT" in line or "STOCKOUT" in line:
            st.error(line)
        elif "FAILED" in line:
            st.warning(line)
        else:
            st.write(line)

    if st.session_state.action_history:
        with st.expander("🗂 Action history", expanded=False):
            st.dataframe(
                pd.DataFrame(st.session_state.action_history),
                use_container_width=True,
                hide_index=True,
            )

# ── Right column: action panel ──────────────────────────────────────

with right:
    st.subheader("🎮 Actions")

    action_type = st.radio(
        "Action type",
        ["move_cargo", "redeploy_stock", "wait"],
        horizontal=True,
        disabled=obs.done,
    )

    if action_type == "move_cargo":
        # Build route options sorted alphabetically
        route_ids = sorted(env.state.edges.keys())
        with st.form("move_cargo_form"):
            shipment_id = st.text_input(
                "shipment_id", value=f"sh-{env.state.step_count + 1}"
            )
            route_id = st.selectbox("route_id", route_ids)
            edge = env.state.edges[route_id] if route_id else None
            if edge:
                st.caption(
                    f"⏱ transit {edge.transit_time}h · "
                    f"💰 ${edge.cost_per_unit:.2f}/unit · "
                    f"🚚 {edge.carrier_type.value} · "
                    f"{'🟢 active' if edge.active and edge.transit_time < 999 else '🔴 disrupted'}"
                )
            rationale = st.text_input(
                "rationale", value="UI-driven move"
            )
            if st.form_submit_button("🚀 Send", use_container_width=True):
                action = MoveCargo(
                    shipment_id=shipment_id,
                    route_id=route_id,
                    carrier_type=edge.carrier_type if edge else CarrierType.SEA,
                    rationale=rationale,
                )
                _apply_action(action)
                st.rerun()

    elif action_type == "redeploy_stock":
        cities = sorted(env.state.nodes.keys())
        with st.form("redeploy_form"):
            from_city = st.selectbox("from_city", cities, index=0)
            to_city = st.selectbox(
                "to_city", cities, index=min(1, len(cities) - 1)
            )
            qty = st.number_input("qty", min_value=1, value=50, step=10)
            rationale = st.text_input("rationale", value="UI-driven redeploy")
            if st.form_submit_button("🔁 Send", use_container_width=True):
                action = RedeployStock(
                    from_city=from_city,
                    to_city=to_city,
                    qty=int(qty),
                    rationale=rationale,
                )
                _apply_action(action)
                st.rerun()

    else:  # wait
        with st.form("wait_form"):
            hours = st.number_input("hours", min_value=1, value=4, step=1)
            rationale = st.text_input("rationale", value="UI-driven wait")
            if st.form_submit_button("⏸ Send", use_container_width=True):
                _apply_action(Wait(hours=int(hours), rationale=rationale))
                st.rerun()

    st.markdown("---")
    st.subheader("📊 Grader detail")
    st.json(report, expanded=False)

    with st.expander("ℹ️ Observation metadata (info)", expanded=False):
        st.json(obs.metadata or {})
