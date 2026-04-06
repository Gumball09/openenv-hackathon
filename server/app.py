"""
FastAPI application for the E-Commerce Logistics Crisis Manager Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from e

try:
    from ..models import (
        LogisticsCrisisManagerObservation,
        MoveCargo,
        RedeployStock,
        Wait,
    )
    from .environment import LogisticsEnv
except ImportError:
    from models import (
        LogisticsCrisisManagerObservation,
        MoveCargo,
        RedeployStock,
        Wait,
    )
    from server.environment import LogisticsEnv

from typing import Union

# The action type for the app is the union of all action types
LogisticsCrisisManagerAction = Union[MoveCargo, RedeployStock, Wait]

app = create_app(
    LogisticsEnv,
    LogisticsCrisisManagerAction,
    LogisticsCrisisManagerObservation,
    env_name="logistics_crisis_manager",
    max_concurrent_envs=4,
)


def main():
    """Entry point for direct execution.

    Honours ``--host`` / ``--port`` CLI flags as well as the
    ``LCM_HOST`` / ``LCM_PORT`` environment variables.
    """
    import argparse
    import os

    import uvicorn

    parser = argparse.ArgumentParser(description="Run Logistics Crisis Manager server")
    parser.add_argument("--host", type=str, default=os.environ.get("LCM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LCM_PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
