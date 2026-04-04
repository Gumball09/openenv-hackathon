"""
Data models for the E-Commerce Logistics Crisis Manager Environment.

Defines state, action, and observation types for a supply-chain
network where an RL agent must move cargo, redeploy stock, and
react to disruptions (port strikes, viral demand, fuel surges, cyber attacks).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Union

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ── Shared sub-models ───────────────────────────────────────────────


class CarrierType(str, Enum):
    AIR = "Air"
    SEA = "Sea"
    RAIL = "Rail"


class Node(State):
    """A warehouse / distribution-centre node in the logistics network."""

    city: str = Field(..., description="City where the node is located")
    capacity: int = Field(..., ge=0, description="Max SKU storage capacity")
    throughput: int = Field(
        ..., ge=0, description="Max SKUs that can be processed per time-step"
    )


class Edge(State):
    """A directed transit link between two nodes."""

    origin: str = Field(..., description="Origin city")
    destination: str = Field(..., description="Destination city")
    transit_time: int = Field(..., ge=0, description="Transit duration in hours")
    cost_per_unit: float = Field(..., ge=0, description="Shipping cost per SKU in USD")
    carrier_type: CarrierType = Field(..., description="Carrier operating this route")
    active: bool = Field(default=True, description="Whether the route is operational")


# ── State ───────────────────────────────────────────────────────────


class LogisticsCrisisManagerState(State):
    """Full internal state of the logistics network."""

    nodes: Dict[str, Node] = Field(
        default_factory=dict,
        description="Warehouse metadata keyed by city name",
    )
    edges: Dict[str, Edge] = Field(
        default_factory=dict,
        description="Transit links keyed by edge ID (e.g. 'Shanghai->LA|Sea')",
    )
    inventory: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="SKU counts per city: {city: {sku: qty}}",
    )
    current_time: int = Field(
        default=0,
        ge=0,
        description="Current simulation clock (hours since start)",
    )
    budget: float = Field(
        default=0.0,
        description="Remaining budget in USD",
    )
    news_feed: List[str] = Field(
        default_factory=list,
        description="Chronological disruption / crisis bulletins",
    )


# ── Actions ─────────────────────────────────────────────────────────


class MoveCargo(Action):
    """Dispatch a shipment along a specific route."""

    type: Literal["move_cargo"] = "move_cargo"
    shipment_id: str = Field(..., description="Unique shipment identifier")
    route_id: str = Field(..., description="ID of the Edge / route to use")
    carrier_type: CarrierType = Field(..., description="Carrier for this shipment")
    rationale: str = Field(
        default="",
        description="Why this move was chosen (e.g. 'Switching to Air because Port is struck')",
    )


class RedeployStock(Action):
    """Transfer inventory between two cities."""

    type: Literal["redeploy_stock"] = "redeploy_stock"
    from_city: str = Field(..., description="Source city")
    to_city: str = Field(..., description="Destination city")
    qty: int = Field(..., gt=0, description="Number of SKUs to transfer")
    rationale: str = Field(
        default="",
        description="Why this redeployment was chosen",
    )


class Wait(Action):
    """Do nothing and let time advance."""

    type: Literal["wait"] = "wait"
    hours: int = Field(..., gt=0, description="Hours to wait")
    rationale: str = Field(
        default="",
        description="Why waiting was chosen over acting",
    )


LogisticsCrisisManagerAction = Annotated[
    Union[MoveCargo, RedeployStock, Wait],
    Field(discriminator="type"),
]


# ── Observation ─────────────────────────────────────────────────────


class LogisticsCrisisManagerObservation(Observation):
    """Human-readable snapshot of the network after an action."""

    summary: str = Field(
        default="",
        description="Descriptive text of the current network status",
    )
    active_delays: List[str] = Field(
        default_factory=list,
        description="Currently active delay or disruption descriptions",
    )
    active_crises: List[str] = Field(
        default_factory=list,
        description="Currently active crisis descriptions",
    )
