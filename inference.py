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
from typing import Any, Dict, List, Optional, Union

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

BENCHMARK = os.environ.get("LCM_BENCHMARK", "logistics_crisis_manager")
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("LCM_SUCCESS_THRESHOLD", "0.5"))

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
#
# The hackathon evaluator parses three line types in this exact format:
#
#   [START] task=<task_name> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
#
# Rules: reward/score formatted to 2 decimal places, booleans lowercase,
# error is the raw message or the literal "null", every record on a
# single line with no embedded newlines.


def _sanitize(value: str) -> str:
    """Collapse whitespace and strip newlines so a field stays single-line."""
    if value is None:
        return "null"
    return " ".join(str(value).split())


def log_start(task: str, env: str, model: str) -> None:
    sys.stdout.write(f"[START] task={task} env={env} model={model}\n")
    sys.stdout.flush()


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err = _sanitize(error) if error else "null"
    sys.stdout.write(
        f"[STEP] step={step} action={_sanitize(action)} "
        f"reward={reward:.2f} done={str(bool(done)).lower()} error={err}\n"
    )
    sys.stdout.flush()


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    sys.stdout.write(
        f"[END] success={str(bool(success)).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}\n"
    )
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
    cfg = TASK_CONFIGS[task_id]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id, seed=0)

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
        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if obs.done:
                break

            step_error: Optional[str] = None
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
                step_error = str(exc)
                sys.stderr.write(f"LLM call failed at step {step}: {exc}\n")
                if consecutive_failures >= 3:
                    log_step(
                        step=step,
                        action="noop",
                        reward=0.0,
                        done=True,
                        error=step_error,
                    )
                    rewards.append(0.0)
                    steps_taken = step
                    break
                content = '{"type":"wait","hours":4,"rationale":"LLM call failed"}'

            action = _parse_action(content, step)
            messages.append({"role": "assistant", "content": content})

            obs = env.step(action)
            messages.append(_obs_to_user_message(obs))

            reward = float(obs.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            action_str = action.model_dump_json(exclude={"metadata"})
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=bool(obs.done),
                error=step_error,
            )

        report = env.grader.report()
        score = float(report["verification_score"])
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        last_error = str(exc)
        sys.stderr.write(f"Task {task_id} crashed: {exc}\n")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
        "error": last_error,
    }


# ── Entry point ─────────────────────────────────────────────────────


def main() -> int:
    if not HF_TOKEN:
        sys.stderr.write(
            "ERROR: HF_TOKEN (or OPENAI_API_KEY) is not set in the environment.\n"
        )
        return 2

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    started_at = time.time()
    for task_id in TASK_CONFIGS:
        run_task(client, task_id)

    sys.stderr.write(
        f"All tasks finished in {time.time() - started_at:.2f}s\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
