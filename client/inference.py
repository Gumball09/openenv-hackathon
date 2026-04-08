"""
E-Commerce Logistics Crisis Manager — OpenAI-compatible inference baseline.

This script runs an LLM agent against all three tiered tasks
(``easy``, ``medium``, ``hard``) using the **standard ``openai`` Python
SDK** pointed at the Hugging Face Inference Router.

Required environment variables (per hackathon submission spec):

    API_BASE_URL  default: https://api-inference.huggingface.co/v1/
    MODEL_NAME    default: Qwen/Qwen2.5-7B-Instruct
    HF_TOKEN      no default — must be provided

Structured stdout logging:

    START
    STEP 1: {"type": "...", ...}
    STEP 2: {"type": "...", ...}
    ...
    END
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple, Union

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' package is required.  Install with: pip install openai"
    ) from e

from ..models import (
    CarrierType,
    LogisticsCrisisManagerObservation,
    MoveCargo,
    RedeployStock,
    Wait,
)
from ..server.environment import TASK_CONFIGS, LogisticsEnv

logger = logging.getLogger(__name__)

# ── Hackathon-mandated environment variables ────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_AGENT_STEPS = 30

SYSTEM_PROMPT = """\
You are a Global Supply-Chain Crisis Manager controlling a 5-node
logistics network (Shanghai, Rotterdam, London, Los Angeles, New York).

You will be told which task you are running:
  - EASY:   routine logistics, no disruptions.
  - MEDIUM: an LA Port Strike fires at step 5 — switch to Air/Rail.
  - HARD:   Port Strike (5) + Fuel Surge (10) + NY Viral Trend (15).

Each turn, output **exactly one JSON action object** — no commentary:

  {"type":"move_cargo","shipment_id":"<id>","route_id":"<Origin>-><Dest>|<Carrier>","carrier_type":"Air|Sea|Rail","rationale":"<why>"}
  {"type":"redeploy_stock","from_city":"<city>","to_city":"<city>","qty":<int>,"rationale":"<why>"}
  {"type":"wait","hours":<int>,"rationale":"<why>"}

Carrier trade-offs:
  Sea  — 48 h, $2/SKU   (cheapest, but blocked when ports strike)
  Rail — 24 h, $4/SKU   (balanced)
  Air  — 12 h, $10/SKU  (emergency reroutes)

Always read the News Feed.  Pre-position stock before crises hit.
"""


# ── Action parsing ──────────────────────────────────────────────────


def _parse_action(
    raw: str, step: int
) -> Tuple[Union[MoveCargo, RedeployStock, Wait], Dict[str, Any]]:
    """Return the parsed action plus the JSON dict that produced it."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Step %d: could not parse JSON, defaulting to Wait", step)
        fallback = {"type": "wait", "hours": 4, "rationale": "Fallback: unparseable LLM output"}
        return Wait(hours=4, rationale=fallback["rationale"]), fallback

    action_json = dict(data)
    kind = data.pop("type", "wait")
    rationale = data.get("rationale", "")
    try:
        if kind == "move_cargo":
            return (
                MoveCargo(
                    shipment_id=data.get("shipment_id", f"auto-{step}"),
                    route_id=data["route_id"],
                    carrier_type=CarrierType(data["carrier_type"]),
                    rationale=rationale,
                ),
                action_json,
            )
        if kind == "redeploy_stock":
            return (
                RedeployStock(
                    from_city=data["from_city"],
                    to_city=data["to_city"],
                    qty=int(data["qty"]),
                    rationale=rationale,
                ),
                action_json,
            )
        if kind == "wait":
            return Wait(hours=int(data.get("hours", 4)), rationale=rationale), action_json
    except (KeyError, ValueError) as exc:
        logger.warning("Step %d: malformed action (%s) — defaulting to Wait", step, exc)
    fallback = {"type": "wait", "hours": 4, "rationale": f"Fallback: malformed '{kind}'"}
    return Wait(hours=4, rationale=fallback["rationale"]), fallback


def _observation_to_user_message(
    obs: LogisticsCrisisManagerObservation,
) -> Dict[str, str]:
    parts = [obs.summary]
    if obs.active_crises:
        parts.append("\nActive Crises:\n" + "\n".join(f"  - {c}" for c in obs.active_crises))
    parts.append(f"\nReward this step: {obs.reward}")
    if obs.done:
        parts.append("\n** EPISODE FINISHED **")
    return {"role": "user", "content": "\n".join(parts)}


# ── Single-task runner ──────────────────────────────────────────────


def run_task(
    client: OpenAI,
    model: str,
    task_id: str,
    step_counter: List[int],
    max_steps: int = MAX_AGENT_STEPS,
) -> Dict[str, Any]:
    """Run the LLM agent against a single tiered task and return its score."""
    env = LogisticsEnv()
    obs = env.reset(task_id=task_id, seed=0)

    cfg = TASK_CONFIGS[task_id]
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"=== TASK: {task_id.upper()} ===\n"
                f"{cfg['description']}\n"
                f"Delivery target: {cfg['delivery_target']} units. "
                f"Min on-time rate: {cfg['min_on_time_rate']:.0%}.\n"
            ),
        },
        _observation_to_user_message(obs),
    ]

    consecutive_failures = 0
    for _ in range(1, max_steps + 1):
        if obs.done:
            break

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=400,
            )
            content = completion.choices[0].message.content or ""
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            logger.error("LLM call failed: %s", exc)
            if consecutive_failures >= 3:
                raise RuntimeError(
                    f"Aborting task '{task_id}' after {consecutive_failures} "
                    f"consecutive LLM failures: {exc}"
                ) from exc
            content = '{"type":"wait","hours":4,"rationale":"LLM call failed"}'

        step_counter[0] += 1
        action, action_json = _parse_action(content, step_counter[0])
        print(
            f"STEP {step_counter[0]}: {json.dumps(action_json, separators=(',', ':'))}",
            flush=True,
        )

        messages.append({"role": "assistant", "content": content})
        obs = env.step(action)
        messages.append(_observation_to_user_message(obs))

    report = env.grader.report()
    return {
        "task_id": task_id,
        "score": report["verification_score"],
        "report": report,
        "final_state": {
            "budget_remaining": round(env.state.budget, 2),
            "deliveries_total": env._deliveries_total,
            "deliveries_on_time": env._deliveries_on_time,
        },
    }


# ── Main: iterate all 3 tasks ───────────────────────────────────────


def _build_client() -> OpenAI:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set.  Export your Hugging Face token before running this script."
        )
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run baseline LLM over all 3 tiered tasks.")
    parser.add_argument("--max-steps", type=int, default=MAX_AGENT_STEPS)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=list(TASK_CONFIGS),
    )
    args = parser.parse_args()

    print("START", flush=True)

    client = _build_client()

    results: List[Dict[str, Any]] = []
    step_counter = [0]
    for task_id in args.tasks:
        try:
            result = run_task(
                client,
                MODEL_NAME,
                task_id,
                step_counter=step_counter,
                max_steps=args.max_steps,
            )
        except Exception:
            logger.exception("Task %s failed", task_id)
            continue
        results.append(result)

    for r in results:
        logger.info("task=%s score=%.4f", r["task_id"], r["score"])

    print("END", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
