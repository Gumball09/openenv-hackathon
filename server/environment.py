"""
E-Commerce Logistics Crisis Manager Environment.

Simulates a 5-node global supply-chain network (backed by a NetworkX
DiGraph) where an RL agent must move cargo, redeploy stock, and react
to four classes of Black-Swan disruptions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import networkx as nx
from openenv.core.env_server.interfaces import Environment

from ..models import (
    CarrierType,
    Edge,
    LogisticsCrisisManagerObservation,
    LogisticsCrisisManagerState,
    MoveCargo,
    Node,
    RedeployStock,
    Wait,
)
from .grader import LogisticsGrader

# ── Constants ───────────────────────────────────────────────────────

TICK_HOURS = 4  # every action advances the clock by this many hours

# Base cost per SKU for Sea carrier (USD).  Air = 5×, Rail = 2×.
SEA_BASE_COST = 2.0

CARRIER_PROFILES: Dict[CarrierType, Dict[str, float]] = {
    CarrierType.SEA: {"transit_hours": 48, "cost_mult": 1.0},
    CarrierType.AIR: {"transit_hours": 12, "cost_mult": 5.0},
    CarrierType.RAIL: {"transit_hours": 24, "cost_mult": 2.0},
}

# City metadata used during reset
CITY_DEFS: List[Dict[str, Any]] = [
    {"city": "Shanghai", "capacity": 5000, "throughput": 800, "role": "Supplier"},
    {"city": "Rotterdam", "capacity": 4000, "throughput": 700, "role": "Supplier"},
    {"city": "New York", "capacity": 3000, "throughput": 600, "role": "High-Demand Hub"},
    {"city": "Los Angeles", "capacity": 3000, "throughput": 600, "role": "High-Demand Hub"},
    {"city": "London", "capacity": 2500, "throughput": 500, "role": "Transit Hub"},
]

# Default edges (every pair that makes geographic sense).
DEFAULT_ROUTES: List[tuple[str, str]] = [
    ("Shanghai", "Los Angeles"),
    ("Shanghai", "Rotterdam"),
    ("Shanghai", "London"),
    ("Rotterdam", "New York"),
    ("Rotterdam", "London"),
    ("London", "New York"),
    ("London", "Los Angeles"),
    ("Los Angeles", "New York"),
]

# Starting SKU inventory per city
DEFAULT_INVENTORY: Dict[str, Dict[str, int]] = {
    "Shanghai": {"electronics": 1200, "textiles": 800},
    "Rotterdam": {"machinery": 600, "chemicals": 400},
    "New York": {"electronics": 100, "machinery": 50},
    "Los Angeles": {"electronics": 80, "textiles": 60},
    "London": {"machinery": 150, "chemicals": 100},
}

STARTING_BUDGET = 100_000.0
MAX_STEPS = 200
ON_TIME_THRESHOLD_H = 48    # deliveries within this window count as "on time"
STOCKOUT_PENALTY = -2.0      # per hub per step with a stockout
LATE_FEE_PER_HOUR = -0.05   # per hour past 48 h threshold
PANIC_TAX_MULT = 0.9        # reward multiplier when over-budget
ON_TIME_MILESTONE = 100      # +1.0 bonus per this many on-time deliveries
FRUGALITY_THRESHOLD = 0.70   # +0.5 bonus if <70 % budget used

# ── Black-swan event triggers ──────────────────────────────────────

PORT_STRIKE_STEP = 5        # LA port strike
VIRAL_TREND_STEP = 15       # NY demand surge
DEMAND_SPIKE_MULT = 5
FUEL_SURGE_MULT = 3         # shipping costs tripled
CYBER_ATTACK_DURATION = 3   # steps with hidden inventory


# ── In-flight shipment tracker ──────────────────────────────────────


@dataclass
class Shipment:
    """A cargo shipment currently in transit."""

    shipment_id: str
    origin: str
    destination: str
    carrier_type: CarrierType
    skus: Dict[str, int]
    depart_time: int
    arrive_time: int


# ── Helpers ─────────────────────────────────────────────────────────


def _make_edge(origin: str, dest: str, carrier: CarrierType) -> Edge:
    """Build an Edge for a given city pair and carrier type."""
    profile = CARRIER_PROFILES[carrier]
    return Edge(
        origin=origin,
        destination=dest,
        transit_time=int(profile["transit_hours"]),
        cost_per_unit=round(SEA_BASE_COST * profile["cost_mult"], 2),
        carrier_type=carrier,
        active=True,
    )


def _edge_id(edge: Edge) -> str:
    """Deterministic ID for an edge."""
    return f"{edge.origin}->{edge.destination}|{edge.carrier_type.value}"


def _build_networkx_graph(edges: Dict[str, Edge]) -> nx.DiGraph:
    """Construct a weighted NetworkX DiGraph from the edge dictionary."""
    G = nx.DiGraph()
    for eid, edge in edges.items():
        G.add_edge(
            edge.origin,
            edge.destination,
            key=eid,
            weight=edge.transit_time,
            cost=edge.cost_per_unit,
            carrier=edge.carrier_type.value,
            active=edge.active,
        )
    return G


# ── Environment ─────────────────────────────────────────────────────


class LogisticsEnv(
    Environment[
        Union[MoveCargo, RedeployStock, Wait],
        LogisticsCrisisManagerObservation,
        LogisticsCrisisManagerState,
    ]
):
    """
    A 5-node global logistics network backed by a NetworkX DiGraph.

    Nodes: Shanghai, Los Angeles, New York, London, Rotterdam.
    The agent can MoveCargo, RedeployStock, or Wait.  Each action
    advances the simulation clock by TICK_HOURS (4 h).

    Black-Swan events:
      1. Port Strike  (step 5)  – LA Sea routes → transit 999 h
      2. Viral Trend  (step 15) – NY demand × 5
      3. Fuel Surge   (random)  – all shipping costs × 3
      4. Cyber Attack  (random)  – inventory hidden for 3 steps
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = LogisticsCrisisManagerState()
        self._shipments: List[Shipment] = []
        self._graph: nx.DiGraph = nx.DiGraph()
        # ── event flags ──
        self._port_strike_triggered: bool = False
        self._viral_trend_triggered: bool = False
        self._fuel_surge_triggered: bool = False
        self._cyber_attack_triggered: bool = False
        self._demand_multiplier: int = 1
        self._cyber_attack_remaining: int = 0
        self._fuel_surge_step: int = 0
        self._cyber_attack_step: int = 0
        # ── reward bookkeeping ──
        self._deliveries_total: int = 0
        self._deliveries_on_time: int = 0
        self._total_spent: float = 0.0
        self._cumulative_late_hours: float = 0.0
        self._rng: random.Random = random.Random()
        # ── programmatic grader ──
        self.grader = LogisticsGrader()

    # ── reset / state ───────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LogisticsCrisisManagerObservation:
        self._reset_rubric()
        self._rng = random.Random(seed)

        eid = episode_id or str(uuid4())

        # Build nodes
        nodes: Dict[str, Node] = {}
        for defn in CITY_DEFS:
            nodes[defn["city"]] = Node(
                city=defn["city"],
                capacity=defn["capacity"],
                throughput=defn["throughput"],
                role=defn["role"],
            )

        # Build edges – one per (route, carrier) combination
        edges: Dict[str, Edge] = {}
        for origin, dest in DEFAULT_ROUTES:
            for carrier in CarrierType:
                edge = _make_edge(origin, dest, carrier)
                edges[_edge_id(edge)] = edge

        # Deep-copy default inventory
        inventory = {city: dict(skus) for city, skus in DEFAULT_INVENTORY.items()}

        self._state = LogisticsCrisisManagerState(
            episode_id=eid,
            step_count=0,
            nodes=nodes,
            edges=edges,
            inventory=inventory,
            current_time=0,
            budget=STARTING_BUDGET,
            news_feed=["Hour 0: Simulation started. All routes operational."],
        )
        self._shipments = []
        self._graph = _build_networkx_graph(edges)

        # ── reset event flags ──
        self._port_strike_triggered = False
        self._viral_trend_triggered = False
        self._fuel_surge_triggered = False
        self._cyber_attack_triggered = False
        self._demand_multiplier = 1
        self._cyber_attack_remaining = 0

        # Pre-roll random event steps (between step 8-25 for fuel, 10-30 for cyber)
        self._fuel_surge_step = self._rng.randint(8, 25)
        self._cyber_attack_step = self._rng.randint(10, 30)

        # ── reset reward bookkeeping ──
        self._deliveries_total = 0
        self._deliveries_on_time = 0
        self._total_spent = 0.0
        self._cumulative_late_hours = 0.0

        # ── reset grader ──
        self.grader.reset()

        return self._observe(reward=0.0)

    @property
    def state(self) -> LogisticsCrisisManagerState:
        return self._state

    # ── step ────────────────────────────────────────────────────────

    def step(
        self,
        action: Union[MoveCargo, RedeployStock, Wait],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LogisticsCrisisManagerObservation:
        s = self._state
        s.step_count += 1

        # --- execute the action ---
        if isinstance(action, MoveCargo):
            self._handle_move_cargo(action)
        elif isinstance(action, RedeployStock):
            self._handle_redeploy_stock(action)
        elif isinstance(action, Wait):
            pass  # nothing to do; clock advances below
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

        # --- advance clock ---
        advance = action.hours if isinstance(action, Wait) else TICK_HOURS
        s.current_time += advance

        # --- black-swan events ---
        self._trigger_events()

        # --- simulate demand drain at high-demand hubs ---
        self._apply_demand_drain()

        # --- deliver arrived shipments ---
        self._process_arrivals()

        # --- tick down cyber-attack visibility blackout ---
        if self._cyber_attack_remaining > 0:
            self._cyber_attack_remaining -= 1

        # --- composite reward ---
        reward = self._compute_reward()

        done = s.step_count >= MAX_STEPS or s.budget <= 0
        obs = self._observe(reward=reward, done=done)

        # --- grader step (records action + observation for scoring) ---
        self.grader.forward(action, obs)

        return obs

    # ── black-swan events ─────────────────────────────────────────

    def _trigger_events(self) -> None:
        """Fire black-swan events at predetermined or random steps."""
        s = self._state

        # ── Step 5: LA Port Strike ──────────────────────────────────
        if s.step_count >= PORT_STRIKE_STEP and not self._port_strike_triggered:
            self._port_strike_triggered = True
            affected = 0
            for eid, edge in s.edges.items():
                is_la_sea = (
                    edge.carrier_type == CarrierType.SEA
                    and ("Los Angeles" in (edge.origin, edge.destination))
                )
                if is_la_sea:
                    edge.transit_time = 999
                    affected += 1
            self._graph = _build_networkx_graph(s.edges)
            s.news_feed.append(
                f"Hour {s.current_time}: ALERT: LA Port Strike in progress! "
                f"All Sea routes to/from Los Angeles blocked "
                f"(transit → 999 h). {affected} routes affected."
            )

        # ── Step 15: NY Viral Trend ─────────────────────────────────
        if s.step_count >= VIRAL_TREND_STEP and not self._viral_trend_triggered:
            self._viral_trend_triggered = True
            self._demand_multiplier = DEMAND_SPIKE_MULT
            self.grader.activate_viral_trend()
            s.news_feed.append(
                f"Hour {s.current_time}: ALERT: Viral Trend in New York! "
                f"Demand surged to {DEMAND_SPIKE_MULT}x normal. "
                f"Inventory will deplete rapidly."
            )

        # ── Random: Fuel Surge ──────────────────────────────────────
        if s.step_count >= self._fuel_surge_step and not self._fuel_surge_triggered:
            self._fuel_surge_triggered = True
            for edge in s.edges.values():
                edge.cost_per_unit = round(edge.cost_per_unit * FUEL_SURGE_MULT, 2)
            self._graph = _build_networkx_graph(s.edges)
            s.news_feed.append(
                f"Hour {s.current_time}: ALERT: Global Fuel Surge! "
                f"All shipping costs tripled ({FUEL_SURGE_MULT}x multiplier)."
            )

        # ── Random: Cyber Attack ────────────────────────────────────
        if s.step_count >= self._cyber_attack_step and not self._cyber_attack_triggered:
            self._cyber_attack_triggered = True
            self._cyber_attack_remaining = CYBER_ATTACK_DURATION
            s.news_feed.append(
                f"Hour {s.current_time}: ALERT: Cyber Attack detected! "
                f"Inventory tracking systems offline for "
                f"{CYBER_ATTACK_DURATION} steps. Data may be unreliable."
            )

    def _apply_demand_drain(self) -> None:
        """Simulate consumer demand consuming inventory at high-demand hubs."""
        s = self._state
        base_drain_per_tick = 10

        for city, node in s.nodes.items():
            if getattr(node, "role", None) != "High-Demand Hub":
                continue
            city_inv = s.inventory.get(city, {})
            drain = base_drain_per_tick * self._demand_multiplier
            remaining_drain = drain
            for sku in list(city_inv.keys()):
                if remaining_drain <= 0:
                    break
                consumed = min(city_inv[sku], remaining_drain)
                city_inv[sku] -= consumed
                remaining_drain -= consumed
            # Track demand ticks for NY during viral trend
            if city == "New York":
                self.grader.record_ny_demand_tick()

    # ── action handlers ─────────────────────────────────────────────

    def _handle_move_cargo(self, action: MoveCargo) -> None:
        s = self._state
        edge = s.edges.get(action.route_id)
        if edge is None:
            s.news_feed.append(
                f"Hour {s.current_time}: MoveCargo FAILED – unknown route "
                f"'{action.route_id}'."
            )
            self.grader.record_invalid_move()
            return

        if not edge.active:
            s.news_feed.append(
                f"Hour {s.current_time}: MoveCargo FAILED – route "
                f"'{action.route_id}' is currently inactive."
            )
            self.grader.record_invalid_move()
            return

        # Blocked-node check: if agent ships Sea to struck LA, record it
        if edge.transit_time >= 999:
            self.grader.record_blocked_node_attempt()

        # Cyber-attack check: agent may be guessing at inventory levels
        origin_inv = s.inventory.get(edge.origin, {})
        total_units = sum(origin_inv.values())
        if self._cyber_attack_remaining > 0:
            self.grader.record_cyber_action(overcount=(total_units == 0))

        if total_units == 0:
            s.news_feed.append(
                f"Hour {s.current_time}: MoveCargo FAILED – no inventory "
                f"at {edge.origin}."
            )
            self.grader.record_invalid_move()
            return

        cost = total_units * edge.cost_per_unit
        if cost > s.budget:
            s.news_feed.append(
                f"Hour {s.current_time}: MoveCargo FAILED – insufficient "
                f"budget (need ${cost:,.2f}, have ${s.budget:,.2f})."
            )
            self.grader.record_invalid_move()
            return

        # Deduct budget, remove inventory, create shipment
        s.budget -= cost
        self._total_spent += cost
        self.grader.record_spending(cost)
        shipped_skus = dict(origin_inv)
        s.inventory[edge.origin] = {sku: 0 for sku in origin_inv}

        self._shipments.append(
            Shipment(
                shipment_id=action.shipment_id,
                origin=edge.origin,
                destination=edge.destination,
                carrier_type=edge.carrier_type,
                skus=shipped_skus,
                depart_time=s.current_time,
                arrive_time=s.current_time + edge.transit_time,
            )
        )

        s.news_feed.append(
            f"Hour {s.current_time}: Shipment '{action.shipment_id}' "
            f"dispatched {edge.origin} → {edge.destination} via "
            f"{edge.carrier_type.value} ({total_units} SKUs, ${cost:,.2f}). "
            f"ETA hour {s.current_time + edge.transit_time}."
        )

    def _handle_redeploy_stock(self, action: RedeployStock) -> None:
        s = self._state

        if action.from_city not in s.nodes:
            s.news_feed.append(
                f"Hour {s.current_time}: RedeployStock FAILED – "
                f"unknown city '{action.from_city}'."
            )
            self.grader.record_invalid_move()
            return
        if action.to_city not in s.nodes:
            s.news_feed.append(
                f"Hour {s.current_time}: RedeployStock FAILED – "
                f"unknown city '{action.to_city}'."
            )
            self.grader.record_invalid_move()
            return

        # Verify reachability via NetworkX
        if not nx.has_path(self._graph, action.from_city, action.to_city):
            s.news_feed.append(
                f"Hour {s.current_time}: RedeployStock FAILED – "
                f"no route from {action.from_city} to {action.to_city}."
            )
            self.grader.record_invalid_move()
            return

        origin_inv = s.inventory.get(action.from_city, {})
        total_available = sum(origin_inv.values())
        if action.qty > total_available:
            s.news_feed.append(
                f"Hour {s.current_time}: RedeployStock FAILED – "
                f"requested {action.qty} SKUs but only {total_available} "
                f"available at {action.from_city}."
            )
            self.grader.record_invalid_move()
            return

        # Withdraw proportionally across SKUs
        remaining = action.qty
        moved: Dict[str, int] = {}
        for sku, count in origin_inv.items():
            take = min(count, remaining)
            if take == 0:
                continue
            origin_inv[sku] -= take
            moved[sku] = take
            remaining -= take
            if remaining == 0:
                break

        # Credit destination
        dest_inv = s.inventory.setdefault(action.to_city, {})
        for sku, qty in moved.items():
            dest_inv[sku] = dest_inv.get(sku, 0) + qty

        s.news_feed.append(
            f"Hour {s.current_time}: Redeployed {action.qty} SKUs "
            f"{action.from_city} → {action.to_city}."
        )

    # ── shipment processing ─────────────────────────────────────────

    def _process_arrivals(self) -> None:
        s = self._state
        still_in_transit: List[Shipment] = []

        for shipment in self._shipments:
            if s.current_time >= shipment.arrive_time:
                dest_inv = s.inventory.setdefault(shipment.destination, {})
                delivered = 0
                for sku, qty in shipment.skus.items():
                    dest_inv[sku] = dest_inv.get(sku, 0) + qty
                    delivered += qty

                transit_duration = shipment.arrive_time - shipment.depart_time
                is_on_time = transit_duration <= ON_TIME_THRESHOLD_H
                self._deliveries_total += 1
                if is_on_time:
                    self._deliveries_on_time += 1
                else:
                    late_hours = transit_duration - ON_TIME_THRESHOLD_H
                    self._cumulative_late_hours += late_hours

                self.grader.record_delivery(
                    on_time=is_on_time, destination=shipment.destination
                )

                on_time_tag = (
                    "ON TIME" if transit_duration <= ON_TIME_THRESHOLD_H else "LATE"
                )
                s.news_feed.append(
                    f"Hour {s.current_time}: Shipment '{shipment.shipment_id}' "
                    f"arrived at {shipment.destination} ({delivered} SKUs, "
                    f"{transit_duration}h transit – {on_time_tag})."
                )
            else:
                still_in_transit.append(shipment)

        self._shipments = still_in_transit

    # ── reward computation ───────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Multi-objective reward with penalties, bonuses, and panic tax.

        Base   = 0.7 × OnTimeRate  +  0.3 × BudgetEfficiency     ∈ [0, 1]

        Penalties (additive):
          • Stockout:   −2.0 per hub at zero inventory with pending orders
          • Late Fee:   −0.05 per hour past 48 h (cumulative this step)

        Bonuses (additive):
          • Milestone:  +1.0 per 100 on-time deliveries
          • Frugality:  +0.5 if < 70 % budget used

        Panic Tax (multiplicative):
          If total_spent > budget, multiply final reward by 0.9.
        """
        # ── on-time rate ────────────────────────────────────────────
        if self._deliveries_total == 0:
            on_time_rate = 1.0
        else:
            on_time_rate = self._deliveries_on_time / self._deliveries_total

        # ── budget efficiency ��──────────────────────────────────────
        budget_efficiency = max(1.0 - (self._total_spent / STARTING_BUDGET), 0.0)

        # ── base composite ──────────────────────────────────────────
        composite = 0.7 * on_time_rate + 0.3 * budget_efficiency

        # ── stockout penalty ────────────────────────────────────────
        composite += self._check_stockout_penalty()

        # ── late fee ────────────────────────────────────────────────
        composite += self._cumulative_late_hours * LATE_FEE_PER_HOUR

        # ── milestone bonus: +1.0 per 100 on-time deliveries ───────
        composite += (self._deliveries_on_time // ON_TIME_MILESTONE) * 1.0

        # ── frugality bonus ─────────────────────────────────────────
        fraction_used = self._total_spent / STARTING_BUDGET
        if fraction_used < FRUGALITY_THRESHOLD:
            composite += 0.5

        # ── panic tax ───────────────────────────────────────────────
        if self._total_spent > STARTING_BUDGET:
            composite *= PANIC_TAX_MULT

        return round(composite, 4)

    def _check_stockout_penalty(self) -> float:
        """Return cumulative penalty for hubs at zero inventory with pending orders."""
        s = self._state
        penalty = 0.0

        pending_destinations: set[str] = {sh.destination for sh in self._shipments}

        for city, node in s.nodes.items():
            if getattr(node, "role", None) != "High-Demand Hub":
                continue
            city_inv = s.inventory.get(city, {})
            total_stock = sum(city_inv.values())
            if total_stock <= 0 and city in pending_destinations:
                penalty += STOCKOUT_PENALTY
                s.news_feed.append(
                    f"Hour {s.current_time}: STOCKOUT at {city}! "
                    f"Warehouse empty while orders are pending."
                )

        return penalty

    # ── graph queries (available to subclasses / tests) ─────────────

    def shortest_path(
        self, origin: str, dest: str, weight: str = "weight"
    ) -> List[str]:
        """Return the shortest path between two cities using NetworkX."""
        try:
            return nx.shortest_path(self._graph, origin, dest, weight=weight)
        except nx.NetworkXNoPath:
            return []

    def cheapest_path(self, origin: str, dest: str) -> List[str]:
        """Return the cheapest path between two cities."""
        return self.shortest_path(origin, dest, weight="cost")

    # ── observation builder ─────────────────────────────────────────

    def _observe(
        self,
        reward: float = 0.0,
        done: bool = False,
    ) -> LogisticsCrisisManagerObservation:
        s = self._state
        is_cyber_blackout = self._cyber_attack_remaining > 0

        # Build inventory summary lines
        inv_lines: List[str] = []
        for city in sorted(s.inventory):
            if is_cyber_blackout:
                inv_lines.append(f"  {city}: [DATA UNAVAILABLE – Cyber Attack]")
                continue
            skus = s.inventory[city]
            total = sum(skus.values())
            breakdown = ", ".join(f"{k}: {v}" for k, v in skus.items() if v)
            role = s.nodes[city].role if city in s.nodes else "Unknown"
            inv_lines.append(f"  {city} [{role}]: {total} SKUs ({breakdown})")

        # In-transit summary
        transit_lines: List[str] = []
        for sh in self._shipments:
            remaining_h = sh.arrive_time - s.current_time
            total = sum(sh.skus.values())
            transit_lines.append(
                f"  '{sh.shipment_id}': {sh.origin} → {sh.destination} "
                f"via {sh.carrier_type.value}, {total} SKUs, "
                f"ETA in {remaining_h}h"
            )

        summary_parts = [
            f"=== Hour {s.current_time} | Budget: ${s.budget:,.2f} "
            f"| Step {s.step_count}/{MAX_STEPS} ===",
            "",
            "Inventory:",
            *inv_lines,
        ]
        if transit_lines:
            summary_parts += ["", "In Transit:", *transit_lines]

        # News feed (last 10 entries to keep observations bounded)
        recent_news = s.news_feed[-10:]
        if recent_news:
            summary_parts += ["", "News Feed:"]
            for n in recent_news:
                summary_parts.append(f"  • {n}")

        # Active delays / crises
        active_delays = [
            n for n in s.news_feed if "FAILED" in n or "delay" in n.lower()
        ]
        active_crises = [
            n
            for n in s.news_feed
            if any(
                kw in n.lower()
                for kw in (
                    "crisis", "closure", "alert", "strike",
                    "surge", "cyber", "stockout",
                )
            )
        ]

        return LogisticsCrisisManagerObservation(
            summary="\n".join(summary_parts),
            active_delays=active_delays,
            active_crises=active_crises,
            done=done,
            reward=reward,
        )
