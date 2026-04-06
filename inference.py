"""
Hackathon submission entry point — Logistics Crisis Manager baseline.

Required by the OpenEnv hackathon spec:
  • Lives in the project root.
  • Uses the standard OpenAI Python client.
  • Reads ``API_BASE_URL``, ``MODEL_NAME``, and ``HF_TOKEN`` from the
    environment.
  • Iterates through every task exposed by the environment and emits
    structured ``[START]`` / ``[STEP]`` / ``[END]`` log lines that the
    automated evaluator parses.

Run::

    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export HF_TOKEN=hf_xxx
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Union

# Make `logistics_crisis_manager` importable when running this file
# directly from the project root (e.g. `python inference.py`).
_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'openai' package is required.  Install with: pip install openai"
    ) from exc

from logistics_crisis_manager.models import (
    CarrierType,
    LogisticsCrisisManagerObservation,
    MoveCargo,
    RedeployStock,
    Wait,
)
from logistics_crisis_manager.server.environment import (
    TASK_CONFIGS,
    LogisticsEnv,
)


# ── Config from env vars ────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

MAX_STEPS_PER_TASK = int(os.environ.get("MAX_STEPS_PER_TASK", "20"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "350"))

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
  Sea  — 48 h, $2/SKU   (cheapest, blocked when ports strike)
  Rail — 24 h, $4/SKU   (balanced)
  Air  — 12 h, $10/SKU  (emergency reroute)

Always read the News Feed.  Pre-position stock before crises hit.
"""


# ── Structured logging helpers ──────────────────────────────────────


def _emit(tag: str, payload: Dict[str, Any]) -> None:
    """Emit a single structured stdout line: ``[TAG] {json}``.

    The hackathon evaluator scrapes lines starting with ``[START]``,
    ``[STEP]``, and ``[END]`` and parses the trailing JSON object.
    Field names and ordering are stable across runs.
    """
    sys.stdout.write(f"[{tag}] {json.dumps(payload, separators=(',', ':'), default=str)}\n")
    sys.stdout.flush()


# ── Action parsing ──────────────────────────────────────────────────


def _parse_action(raw: str, step: int) -> Union[MoveCargo, RedeployStock, Wait]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Wait(hours=4, rationale="Fallback: unparseable LLM output")

    kind = data.pop("type", "wait")
    rationale = data.get("rationale", "")
    try:
        if kind == "move_cargo":
            return MoveCargo(
                shipment_id=data.get("shipment_id", f"auto-{step}"),
                route_id=data["route_id"],
                carrier_type=CarrierType(data["carrier_type"]),
                rationale=rationale,
            )
        if kind == "redeploy_stock":
            return RedeployStock(
                from_city=data["from_city"],
                to_city=data["to_city"],
                qty=int(data["qty"]),
                rationale=rationale,
            )
        if kind == "wait":
            return Wait(hours=int(data.get("hours", 4)), rationale=rationale)
    except (KeyError, ValueError) as exc:
        return Wait(hours=4, rationale=f"Fallback: malformed '{kind}' ({exc})")
    return Wait(hours=4, rationale=f"Fallback: unknown action '{kind}'")


def _obs_to_user_message(obs: LogisticsCrisisManagerObservation) -> Dict[str, str]:
    parts = [obs.summary]
    if obs.active_crises:
        parts.append("\nActive Crises:\n" + "\n".join(f"  - {c}" for c in obs.active_crises))
    parts.append(f"\nReward this step: {obs.reward}")
    if obs.done:
        parts.append("\n** EPISODE FINISHED **")
    return {"role": "user", "content": "\n".join(parts)}


# ── Per-task runner ─────────────────────────────────────────────────


def run_task(client: OpenAI, task_id: str) -> Dict[str, Any]:
    env = LogisticsEnv()
    obs = env.reset(task_id=task_id, seed=0)
    cfg = TASK_CONFIGS[task_id]

    _emit(
        "START",
        {
            "task_id": task_id,
            "model": MODEL_NAME,
            "api_base_url": API_BASE_URL,
            "max_steps": MAX_STEPS_PER_TASK,
            "delivery_target": cfg["delivery_target"],
            "min_on_time_rate": cfg["min_on_time_rate"],
            "description": cfg["description"],
        },
    )

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
        _obs_to_user_message(obs),
    ]

    consecutive_failures = 0
    last_error: str = ""
    for step in range(1, MAX_STEPS_PER_TASK + 1):
        if obs.done:
            break

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            content = completion.choices[0].message.content or ""
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            last_error = str(exc)
            sys.stderr.write(f"LLM call failed at step {step}: {exc}\n")
            if consecutive_failures >= 3:
                break
            content = '{"type":"wait","hours":4,"rationale":"LLM call failed"}'

        action = _parse_action(content, step)
        messages.append({"role": "assistant", "content": content})

        obs = env.step(action)
        messages.append(_obs_to_user_message(obs))

        _emit(
            "STEP",
            {
                "task_id": task_id,
                "step": step,
                "action_type": action.type,
                "action": json.loads(action.model_dump_json(exclude={"metadata"})),
                "reward": obs.reward,
                "done": obs.done,
                "info": obs.metadata or {},
            },
        )

    report = env.grader.report()
    end_payload = {
        "task_id": task_id,
        "score": report["verification_score"],
        "safety": report["safety"],
        "reliability": report["reliability"],
        "efficiency": report["efficiency"],
        "task_progress": report["task_progress"],
        "deliveries_total": env._deliveries_total,
        "deliveries_on_time": env._deliveries_on_time,
        "budget_remaining": round(env.state.budget, 2),
        "steps_taken": env.state.step_count,
    }
    if last_error and consecutive_failures >= 3:
        end_payload["aborted"] = True
        end_payload["error"] = last_error
    _emit("END", end_payload)
    return end_payload


# ── Entry point ─────────────────────────────────────────────────────


def main() -> int:
    if not HF_TOKEN:
        sys.stderr.write(
            "ERROR: HF_TOKEN (or OPENAI_API_KEY) is not set in the environment.\n"
        )
        return 2

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    started_at = time.time()
    results: List[Dict[str, Any]] = []
    for task_id in TASK_CONFIGS:
        try:
            results.append(run_task(client, task_id))
        except Exception as exc:  # pragma: no cover - defensive
            sys.stderr.write(f"Task {task_id} crashed: {exc}\n")
            _emit("END", {"task_id": task_id, "score": 0.0, "error": str(exc)})

    avg = (
        sum(r.get("score", 0.0) for r in results) / len(results)
        if results
        else 0.0
    )
    _emit(
        "SUMMARY",
        {
            "tasks": [r["task_id"] for r in results],
            "scores": {r["task_id"]: r.get("score", 0.0) for r in results},
            "average_score": round(avg, 4),
            "elapsed_seconds": round(time.time() - started_at, 2),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
