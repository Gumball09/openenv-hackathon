"""
Programmatic Grader for the E-Commerce Logistics Crisis Manager.

Computes a Verification Score from three sub-scores:
  • Safety      – penalises invalid moves (shipping to blocked nodes, etc.)
  • Reliability – measures fulfilment of mandatory Viral-Trend orders
  • Efficiency  – the 0.7 × OnTimeRate + 0.3 × BudgetEfficiency formula

The grader is a standard openenv Rubric and can be wired directly into
the Environment or invoked independently after an episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from openenv.core.rubrics import Rubric

try:
    from ..models import (
        LogisticsCrisisManagerObservation,
        MoveCargo,
        RedeployStock,
        Wait,
    )
except ImportError:  # standalone Docker layout: server/ is top-level
    from models import (
        LogisticsCrisisManagerObservation,
        MoveCargo,
        RedeployStock,
        Wait,
    )


@dataclass
class GraderStats:
    """Running statistics accumulated across the episode."""

    # Per-step safety penalty for the *most recent* step (1.0 = clean)
    last_step_safety: float = 1.0

    # Safety
    total_actions: int = 0
    invalid_moves: int = 0          # FAILED actions (blocked routes, bad cities, …)
    blocked_node_attempts: int = 0  # specifically shipping Sea to struck LA

    # Reliability — Viral-Trend fulfilment
    viral_trend_active: bool = False
    ny_deliveries_during_trend: int = 0
    ny_demand_ticks_during_trend: int = 0

    # Efficiency (mirrors env bookkeeping, but self-contained)
    deliveries_total: int = 0
    deliveries_on_time: int = 0
    total_spent: float = 0.0
    starting_budget: float = 100_000.0

    # Cyber-attack awareness
    actions_during_cyber_blackout: int = 0
    overcount_during_blackout: int = 0  # shipped more than actually available

    # Panic-tax tracking
    budget_overrun: bool = False


class LogisticsGrader(Rubric):
    """Programmatic Verification Grader.

    Produces a float score in [0, 1] computed as::

        score = w_s * safety + w_r * reliability + w_e * efficiency

    Weights default to 0.2 / 0.3 / 0.5 (efficiency-heavy).
    """

    def __init__(
        self,
        weight_safety: float = 0.2,
        weight_reliability: float = 0.3,
        weight_efficiency: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_s = weight_safety
        self.w_r = weight_reliability
        self.w_e = weight_efficiency
        self.stats = GraderStats()
        self._action_log: List[Dict[str, Any]] = []
        # ── per-task scoring config ──
        self.task_id: str = "medium"
        self.task_cfg: Dict[str, Any] = {}

    # ── Per-task config ─────────────────────────────────────────────

    def set_task(self, task_id: str, task_cfg: Dict[str, Any]) -> None:
        """Bind the grader to a specific tiered task scenario."""
        self.task_id = task_id
        self.task_cfg = dict(task_cfg)

    # ── Rubric interface ────────────────────────────────────────────

    def forward(self, action: Any, observation: Any) -> float:
        """Score a single step.  Returns the running verification score.

        Per-step safety: destructive actions (shipping to a blocked
        node, malformed inputs, …) cause `record_*` to drop
        `last_step_safety` *before* this method is called.
        """
        self._record_action(action, observation)
        return self.verification_score()

    def begin_step(self) -> None:
        """Called by the environment at the start of every step."""
        self.stats.last_step_safety = 1.0

    def reset(self) -> None:
        self.stats = GraderStats()
        self._action_log = []

    # ── Recording ───────────────────────────────────────────────────

    def record_invalid_move(self) -> None:
        self.stats.invalid_moves += 1
        # Soft per-step penalty (an honest mistake — not destructive).
        self.stats.last_step_safety = min(self.stats.last_step_safety, 0.5)

    def record_blocked_node_attempt(self) -> None:
        self.stats.blocked_node_attempts += 1
        # Hard per-step penalty: shipping to a blocked node is destructive.
        self.stats.last_step_safety = 0.0

    def record_delivery(self, *, on_time: bool, destination: str) -> None:
        self.stats.deliveries_total += 1
        if on_time:
            self.stats.deliveries_on_time += 1
        if self.stats.viral_trend_active and destination == "New York":
            self.stats.ny_deliveries_during_trend += 1

    def record_spending(self, amount: float) -> None:
        self.stats.total_spent += amount
        if self.stats.total_spent > self.stats.starting_budget:
            self.stats.budget_overrun = True

    def activate_viral_trend(self) -> None:
        self.stats.viral_trend_active = True

    def record_ny_demand_tick(self) -> None:
        if self.stats.viral_trend_active:
            self.stats.ny_demand_ticks_during_trend += 1

    def record_cyber_action(self, *, overcount: bool = False) -> None:
        self.stats.actions_during_cyber_blackout += 1
        if overcount:
            self.stats.overcount_during_blackout += 1

    def _record_action(self, action: Any, observation: Any) -> None:
        """Log action + rationale for later LLM-judge consumption."""
        self.stats.total_actions += 1
        entry: Dict[str, Any] = {"step": self.stats.total_actions}
        if hasattr(action, "rationale"):
            entry["rationale"] = action.rationale
        if hasattr(action, "type"):
            entry["type"] = action.type
        if hasattr(observation, "summary"):
            entry["summary_len"] = len(observation.summary)
        self._action_log.append(entry)

    # ── Sub-scores ──────────────────────────────────────────────────

    def safety_score(self) -> float:
        """1.0 = no invalid moves;  degrades linearly, floor at 0."""
        if self.stats.total_actions == 0:
            return 1.0
        bad = self.stats.invalid_moves + self.stats.blocked_node_attempts
        bad += self.stats.overcount_during_blackout
        return max(1.0 - bad / self.stats.total_actions, 0.0)

    def reliability_score(self) -> float:
        """How well the agent handled Viral-Trend demand at NY.

        If the viral trend never fired, reliability is 1.0 (vacuously true).
        Otherwise scored as fraction of demand ticks that received a delivery.
        """
        st = self.stats
        if not st.viral_trend_active:
            return 1.0
        if st.ny_demand_ticks_during_trend == 0:
            return 1.0
        # Each demand tick ideally has at least one delivery to NY
        rate = st.ny_deliveries_during_trend / st.ny_demand_ticks_during_trend
        return min(rate, 1.0)

    def efficiency_score(self) -> float:
        """0.7 × OnTimeRate  +  0.3 × BudgetEfficiency."""
        st = self.stats
        if st.deliveries_total == 0:
            on_time_rate = 1.0
        else:
            on_time_rate = st.deliveries_on_time / st.deliveries_total
        budget_eff = max(1.0 - st.total_spent / st.starting_budget, 0.0)
        score = 0.7 * on_time_rate + 0.3 * budget_eff
        # Panic-tax: multiply by 0.9 if budget overrun
        if st.budget_overrun:
            score *= 0.9
        return score

    # ── Per-task scoring ───────────────────────────────────────────

    def task_progress_score(self) -> float:
        """Partial-credit progress toward the active task's goal.

        Linearly rewards delivery progress up to `delivery_target` and
        — for tasks with a stricter on-time bar — multiplies by the
        ratio of achieved/required on-time rate (capped at 1.0).
        """
        st = self.stats
        target = float(self.task_cfg.get("delivery_target") or 1)
        delivered_credit = min(st.deliveries_total / target, 1.0)

        # On-time gate
        on_time_rate = (
            st.deliveries_on_time / st.deliveries_total
            if st.deliveries_total > 0
            else 0.0
        )
        required = float(self.task_cfg.get("min_on_time_rate") or 0.0)
        if required > 0:
            on_time_ratio = min(on_time_rate / required, 1.0)
        else:
            on_time_ratio = 1.0

        return round(delivered_credit * on_time_ratio, 4)

    def verification_score(self) -> float:
        """Strict task-aware score in [0, 1].

        Composition:
            score = 0.20 * safety
                  + 0.30 * reliability   (viral-trend handling, etc.)
                  + 0.25 * efficiency    (on-time + budget)
                  + 0.25 * task_progress (toward the tier's target)

        For the **hard** task an additional reliability multiplier is
        applied: failure to deliver into NY during the Viral Trend caps
        the reliability sub-score at 0.4, which dominates the final
        score.  This makes "Hard" genuinely difficult.
        """
        safety = self.safety_score()
        reliability = self.reliability_score()
        efficiency = self.efficiency_score()
        progress = self.task_progress_score()

        # Hard-task strictness: severely penalise unhandled viral trend.
        if self.task_id == "hard" and self.stats.viral_trend_active:
            if self.stats.ny_deliveries_during_trend == 0:
                reliability = min(reliability, 0.2)
                progress *= 0.5

        # Medium-task strictness: missing the 80% on-time bar caps
        # progress.
        if self.task_id == "medium" and self.stats.deliveries_total > 0:
            on_time_rate = (
                self.stats.deliveries_on_time / self.stats.deliveries_total
            )
            if on_time_rate < float(self.task_cfg.get("min_on_time_rate", 0.8)):
                progress *= 0.6

        score = (
            0.20 * safety
            + 0.30 * reliability
            + 0.25 * efficiency
            + 0.25 * progress
        )
        return round(max(min(score, 1.0), 0.0), 4)

    # ── Reporting ───────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Return a human-readable grading report dict."""
        return {
            "task_id": self.task_id,
            "safety": round(self.safety_score(), 4),
            "reliability": round(self.reliability_score(), 4),
            "efficiency": round(self.efficiency_score(), 4),
            "task_progress": self.task_progress_score(),
            "verification_score": self.verification_score(),
            "stats": {
                "total_actions": self.stats.total_actions,
                "invalid_moves": self.stats.invalid_moves,
                "blocked_node_attempts": self.stats.blocked_node_attempts,
                "deliveries_total": self.stats.deliveries_total,
                "deliveries_on_time": self.stats.deliveries_on_time,
                "total_spent": round(self.stats.total_spent, 2),
                "budget_overrun": self.stats.budget_overrun,
                "viral_trend_active": self.stats.viral_trend_active,
                "ny_deliveries_during_trend": self.stats.ny_deliveries_during_trend,
                "ny_demand_ticks_during_trend": self.stats.ny_demand_ticks_during_trend,
                "overcount_during_blackout": self.stats.overcount_during_blackout,
            },
        }

    @property
    def action_log(self) -> List[Dict[str, Any]]:
        """Full action log with rationales — fed to the LLM Judge."""
        return list(self._action_log)
