"""
E-Commerce Logistics Crisis Manager — OpenAI-compatible inference baseline.

This script runs a frontier LLM agent against all three tiered tasks
(``easy``, ``medium``, ``hard``) and prints a reproducible baseline
score for each.

It uses the **standard ``openai`` Python SDK** and reads
``OPENAI_API_KEY`` from the environment.  By default it talks to the
OpenAI API; pass ``--hf`` (or set ``USE_HF_ROUTER=1``) to point at the
Hugging Face Inference Router so hackathon evaluators can run it for
free with an HF token stored in ``OPENAI_API_KEY``.

Usage::

    export OPENAI_API_KEY=sk-...                  # OpenAI
    python -m logistics_crisis_manager.client.inference

    export OPENAI_API_KEY=hf_...                  # HF Router
    python -m logistics_crisis_manager.client.inference --hf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Union

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

# ── Endpoint configuration ──────────────────────────────────────────

# Per the hackathon brief, the HF Router exposes an OpenAI-compatible
# /v1 endpoint at the URL below.  This is selected via --hf or
# USE_HF_ROUTER=1.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

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
) -> Union[MoveCargo, RedeployStock, Wait]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Some models prepend prose; grab the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Step %d: could not parse JSON, defaulting to Wait", step)
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
        logger.warning("Step %d: malformed action (%s) — defaulting to Wait", step, exc)
    return Wait(hours=4, rationale=f"Fallback: malformed '{kind}'")


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
    max_steps: int = MAX_AGENT_STEPS,
    verbose: bool = False,
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
    for step in range(1, max_steps + 1):
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
            logger.error("LLM call failed at step %d: %s", step, exc)
            if consecutive_failures >= 3:
                raise RuntimeError(
                    f"Aborting task '{task_id}' after {consecutive_failures} "
                    f"consecutive LLM failures: {exc}"
                ) from exc
            content = '{"type":"wait","hours":4,"rationale":"LLM call failed"}'

        action = _parse_action(content, step)
        messages.append({"role": "assistant", "content": content})

        obs = env.step(action)
        messages.append(_observation_to_user_message(obs))

        if verbose:
            print(f"  [step {step}] action={action.type} reward={obs.reward}")

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


def _build_client(use_hf: bool) -> tuple[OpenAI, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.  Export your OpenAI key (or HF token "
            "when using --hf) before running this script."
        )
    if use_hf:
        client = OpenAI(api_key=api_key, base_url=HF_ROUTER_BASE_URL)
        model = os.environ.get("LCM_MODEL", DEFAULT_HF_MODEL)
    else:
        client = OpenAI(api_key=api_key)
        model = os.environ.get("LCM_MODEL", DEFAULT_OPENAI_MODEL)
    return client, model


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run baseline LLM over all 3 tiered tasks.")
    parser.add_argument(
        "--hf",
        action="store_true",
        default=os.environ.get("USE_HF_ROUTER") == "1",
        help="Use the Hugging Face Router (OpenAI-compatible) instead of OpenAI.",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_AGENT_STEPS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=list(TASK_CONFIGS),
    )
    args = parser.parse_args()

    client, model = _build_client(args.hf)
    backend = "Hugging Face Router" if args.hf else "OpenAI"
    print(f"Backend: {backend}  |  Model: {model}")
    print("=" * 60)

    results: List[Dict[str, Any]] = []
    for task_id in args.tasks:
        print(f"\n▶ Running task: {task_id.upper()}")
        try:
            result = run_task(
                client, model, task_id,
                max_steps=args.max_steps, verbose=args.verbose,
            )
        except Exception:
            logger.exception("Task %s failed", task_id)
            continue
        results.append(result)
        rep = result["report"]
        st = result["final_state"]
        print(
            f"  → Reproducible Baseline Score [{task_id}] = "
            f"{result['score']:.4f}"
        )
        print(
            f"     safety={rep['safety']:.2f}  reliability={rep['reliability']:.2f}  "
            f"efficiency={rep['efficiency']:.2f}  progress={rep['task_progress']:.2f}"
        )
        print(
            f"     deliveries={st['deliveries_on_time']}/{st['deliveries_total']} on-time, "
            f"budget left=${st['budget_remaining']:,.2f}"
        )

    print("\n" + "=" * 60)
    print("Summary (Reproducible Baseline Scores)")
    print("=" * 60)
    for r in results:
        print(f"  {r['task_id']:>6}  →  {r['score']:.4f}")
    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"  {'avg':>6}  →  {avg:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
