"""
E-Commerce Logistics Crisis Manager – LLM inference client.

Dual-Layer Evaluation:
  1. Programmatic Grader  (LogisticsGrader)  → Verification Score
  2. LLM Judge            (openenv LLMJudge) → Strategic Score
  3. Combined             (0.6 × Programmatic + 0.4 × Strategic) → Submission Score
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from openenv.core.llm_client import LLMResponse, OpenAIClient
from openenv.core.rubrics import LLMJudge

from ..models import (
    CarrierType,
    LogisticsCrisisManagerObservation,
    MoveCargo,
    RedeployStock,
    Wait,
)
from ..server.environment import LogisticsEnv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────

HF_ROUTER_URL = "https://router.huggingface.co"
HF_ROUTER_PORT = 443
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
MAX_AGENT_STEPS = 30

SYSTEM_PROMPT = """\
You are a **Global Supply Chain Crisis Manager** controlling a logistics \
network of five cities: Shanghai, Rotterdam, London, Los Angeles, and \
New York.

Shanghai and Rotterdam are **Supplier** nodes with large inventories. \
New York and Los Angeles are **High-Demand Hubs** that constantly consume \
stock. London is a **Transit Hub**.

## Your Objective
Maximise your composite reward:
  Reward = 0.7 × OnTimeRate + 0.3 × BudgetEfficiency
  • OnTimeRate  = fraction of deliveries completed within 48 hours.
  • BudgetEff   = 1 − (TotalSpent / $100,000 initial budget).

### Bonuses
  • +1.0 for every 100 cumulative on-time deliveries.
  • +0.5 if you use less than 70% of the budget.

### Penalties
  • −2.0 STOCKOUT: any High-Demand Hub at zero inventory with pending orders.
  • −0.05/hr LATE FEE: for every hour a delivery exceeds the 48 h threshold.
  • ×0.9 PANIC TAX: if cumulative spending exceeds the starting budget.

## Carrier Trade-offs (CRITICAL — choose wisely)
| Carrier | Transit | Cost/SKU | Best for |
|---------|---------|----------|----------|
| Sea     | 48 h    | $2.00    | Cheap bulk — only when 48 h window is safe |
| Rail    | 24 h    | $4.00    | Balanced middle ground |
| Air     | 12 h    | $10.00  | Emergencies & crisis resupply (5× Sea cost) |

## Black-Swan Events — Monitor the News Feed!
You MUST read the "News Feed" section in every observation and adapt:

1. **LA Port Strike** (~hour 20) — All Sea routes to/from Los Angeles are \
blocked (transit → 999 h).  Switch to Air or Rail immediately, or reroute \
through London.
2. **NY Viral Trend** (~hour 60) — New York demand surges to 5×.  \
Pre-position stock beforehand or air-freight emergency resupply.
3. **Fuel Surge** (random) — All shipping costs triple overnight.  \
Conserve budget; batch shipments; avoid unnecessary moves.
4. **Cyber Attack** (random) — Inventory tracking goes dark for 3 steps.  \
You will see "[DATA UNAVAILABLE]" instead of stock counts.  Rely on your \
memory of prior observations and in-transit shipments.

## Available Actions — respond with exactly ONE JSON object per turn

Every action MUST include a "rationale" field explaining your reasoning.

1. **move_cargo** — dispatch all inventory at origin along a route.
   ```json
   {"type":"move_cargo","shipment_id":"<unique>","route_id":"<Origin>-><Dest>|<Carrier>","carrier_type":"<Air|Sea|Rail>","rationale":"<why>"}
   ```

2. **redeploy_stock** — instantly transfer SKUs between connected cities.
   ```json
   {"type":"redeploy_stock","from_city":"<city>","to_city":"<city>","qty":<int>,"rationale":"<why>"}
   ```

3. **wait** — do nothing and advance the clock.
   ```json
   {"type":"wait","hours":<int>,"rationale":"<why>"}
   ```

## Rules
- Respond with **only** the JSON action object — no commentary.
- Always include a "rationale" field explaining your strategic thinking.
- Think ahead: pre-position stock before crises hit.
- Minimise unnecessary spending; Sea is cheapest when time allows.
- If a route is blocked, use an alternative carrier or reroute.
"""

# ── Tool definitions (MCP format) ──────────────────────────────────

ACTION_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "move_cargo",
        "description": (
            "Dispatch all inventory at the origin city along a route. "
            "Route ID format: 'Origin->Destination|Carrier'. "
            "WARNING: sending cargo via Sea to a struck port results in "
            "999 h transit and a massive late fee."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "shipment_id": {
                    "type": "string",
                    "description": "Unique shipment identifier",
                },
                "route_id": {
                    "type": "string",
                    "description": "Edge ID, e.g. 'Shanghai->Los Angeles|Sea'",
                },
                "carrier_type": {
                    "type": "string",
                    "enum": ["Air", "Sea", "Rail"],
                    "description": "Carrier for this shipment",
                },
                "rationale": {
                    "type": "string",
                    "description": "Strategic reasoning for this move",
                },
            },
            "required": ["shipment_id", "route_id", "carrier_type", "rationale"],
        },
    },
    {
        "name": "redeploy_stock",
        "description": (
            "Instantly transfer SKUs between two connected cities. "
            "Fails if no graph path exists between them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_city": {"type": "string", "description": "Source city"},
                "to_city": {"type": "string", "description": "Destination city"},
                "qty": {
                    "type": "integer",
                    "description": "Number of SKUs to transfer",
                    "minimum": 1,
                },
                "rationale": {
                    "type": "string",
                    "description": "Strategic reasoning for this redeployment",
                },
            },
            "required": ["from_city", "to_city", "qty", "rationale"],
        },
    },
    {
        "name": "wait",
        "description": "Do nothing and let time advance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Hours to wait",
                    "minimum": 1,
                },
                "rationale": {
                    "type": "string",
                    "description": "Strategic reasoning for waiting",
                },
            },
            "required": ["hours", "rationale"],
        },
    },
]

# ── LLM Judge prompt template ──────────────────────────────────────

LLM_JUDGE_PROMPT = """\
You are an expert logistics evaluator.  Below is the complete trajectory \
of an AI agent managing a global supply-chain crisis simulation.

## News Feed (Crisis Log)
{news_feed}

## Agent Action Log (with rationales)
{action_log}

## Final Statistics
{stats}

## Evaluation Criteria — score each 0.0 to 1.0:

1. **Crisis Awareness** (weight 0.35):
   - Did the agent acknowledge the Port Strike in its rationale?
   - Did it react by switching carriers or rerouting after the strike alert?
   - Did it notice and mention the Fuel Surge / Cyber Attack?

2. **Strategic Redeployment** (weight 0.35):
   - Was inventory redeployment strategic (pre-positioning for demand)?
   - Or was it random / reactive moves with no clear plan?
   - Did the agent prioritize the NY Viral Trend over low-priority orders?

3. **Cost Discipline** (weight 0.30):
   - Did the agent use Sea when safe and Air only in emergencies?
   - Did it avoid wasteful spending after the Fuel Surge?
   - Was overall budget management prudent?

Respond with ONLY a JSON object:
{{"crisis_awareness": <float>, "strategic_redeployment": <float>, \
"cost_discipline": <float>, "overall": <float>, \
"reasoning": "<1-2 sentence explanation>"}}
"""

# ── Action parsing ──────────────────────────────────────────────────


def parse_action(
    response: LLMResponse, step: int
) -> Union[MoveCargo, RedeployStock, Wait]:
    """Convert an LLM response into a typed Action (with rationale)."""
    if response.tool_calls:
        tc = response.tool_calls[0]
        return _build_action(tc.name, tc.args, step)

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM output as JSON, defaulting to Wait(4h)")
        return Wait(hours=4, rationale="Fallback: unparseable LLM output")

    action_type = data.pop("type", "wait")
    return _build_action(action_type, data, step)


def _build_action(
    name: str, args: Dict[str, Any], step: int
) -> Union[MoveCargo, RedeployStock, Wait]:
    rationale = args.get("rationale", "")
    if name == "move_cargo":
        return MoveCargo(
            shipment_id=args.get("shipment_id", f"auto-{step}"),
            route_id=args["route_id"],
            carrier_type=CarrierType(args["carrier_type"]),
            rationale=rationale,
        )
    elif name == "redeploy_stock":
        return RedeployStock(
            from_city=args["from_city"],
            to_city=args["to_city"],
            qty=int(args["qty"]),
            rationale=rationale,
        )
    elif name == "wait":
        return Wait(
            hours=int(args.get("hours", 4)),
            rationale=rationale,
        )
    else:
        logger.warning(f"Unknown action '{name}', defaulting to Wait(4h)")
        return Wait(hours=4, rationale=f"Fallback: unknown action '{name}'")


# ── Observation → message ───────────────────────────────────────────


def observation_to_message(
    obs: LogisticsCrisisManagerObservation,
) -> Dict[str, str]:
    """Format an observation as an OpenAI-style user message."""
    parts = [obs.summary]
    if obs.active_crises:
        parts.append(
            "\nActive Crises:\n"
            + "\n".join(f"  - {c}" for c in obs.active_crises)
        )
    if obs.active_delays:
        parts.append(
            "\nDelays:\n" + "\n".join(f"  - {d}" for d in obs.active_delays)
        )
    parts.append(f"\nReward this step: {obs.reward}")
    if obs.done:
        parts.append("\n** EPISODE FINISHED **")
    return {"role": "user", "content": "\n".join(parts)}


# ── LLM Judge ──────────────────────────────────────────────────────


async def run_llm_judge(
    llm: OpenAIClient,
    env: LogisticsEnv,
) -> Dict[str, Any]:
    """Invoke the LLM Judge on the completed trajectory.

    Passes the full news_feed and action log (with rationales) to the
    judge so it can evaluate crisis awareness and strategic quality.
    """
    grader = env.grader
    grader_report = grader.report()

    # Format inputs for the judge
    news_feed_text = "\n".join(
        f"  {i+1}. {entry}" for i, entry in enumerate(env.state.news_feed)
    )
    action_log_text = "\n".join(
        f"  Step {e.get('step', '?')}: [{e.get('type', '?')}] "
        f"rationale=\"{e.get('rationale', 'N/A')}\""
        for e in grader.action_log
    )
    stats_text = json.dumps(grader_report["stats"], indent=2)

    prompt = LLM_JUDGE_PROMPT.format(
        news_feed=news_feed_text,
        action_log=action_log_text,
        stats=stats_text,
    )

    try:
        raw_response = await llm.complete(prompt, max_tokens=512)
        # Parse the JSON response
        text = raw_response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        judge_result = json.loads(text)
        # Clamp scores to [0, 1]
        for key in ("crisis_awareness", "strategic_redeployment",
                     "cost_discipline", "overall"):
            if key in judge_result:
                judge_result[key] = max(0.0, min(1.0, float(judge_result[key])))
        return judge_result
    except Exception:
        logger.exception("LLM Judge call failed, using default scores")
        return {
            "crisis_awareness": 0.5,
            "strategic_redeployment": 0.5,
            "cost_discipline": 0.5,
            "overall": 0.5,
            "reasoning": "Judge evaluation failed — default scores applied.",
        }


# ── API key resolution ─────────────────────────────────────────────


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve HuggingFace API key from arg → env vars → error."""
    if api_key:
        return api_key
    for var in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN"):
        val = os.environ.get(var, "")
        if val:
            return val
    raise RuntimeError(
        "No API key provided. Set HUGGING_FACE_HUB_TOKEN or HF_TOKEN "
        "in your environment, or pass api_key= to run_agent()."
    )


# ── Main agent loop ─────────────────────────────────────────────────


async def run_agent(
    env: LogisticsEnv | None = None,
    api_key: str | None = None,
    max_steps: int = MAX_AGENT_STEPS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the LLM agent loop, then evaluate with both graders.

    Returns a dict with:
      - observations: list of observations
      - programmatic_score: LogisticsGrader verification score
      - llm_judge_score: LLM Judge strategic score
      - final_score: weighted combination
      - grader_report: full programmatic breakdown
      - judge_report: full LLM judge breakdown
    """
    token = _resolve_api_key(api_key)

    llm = OpenAIClient(
        endpoint=HF_ROUTER_URL,
        port=HF_ROUTER_PORT,
        model=MODEL_ID,
        api_key=token,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=512,
    )

    if env is None:
        env = LogisticsEnv()

    obs = env.reset()
    observations: List[LogisticsCrisisManagerObservation] = [obs]

    # Conversation history (multi-turn)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        observation_to_message(obs),
    ]

    for step in range(1, max_steps + 1):
        if obs.done:
            break

        # ── LLM decision ────────────────────────────────────────────
        try:
            response = await llm.complete_with_tools(
                messages=messages,
                tools=ACTION_TOOLS,
            )
        except Exception:
            logger.exception("LLM call failed at step %d, falling back to Wait", step)
            response = LLMResponse(
                content='{"type":"wait","hours":4,"rationale":"LLM call failed"}'
            )

        action = parse_action(response, step)

        if verbose:
            action_summary = action.model_dump_json(exclude={"metadata"})
            print(f"\n[Step {step}] Action: {action_summary}")

        messages.append(response.to_message_dict())

        # ── Environment step ─────────────────────────────────────────
        obs = env.step(action)
        observations.append(obs)

        if verbose:
            print(obs.summary)
            if obs.active_crises:
                for c in obs.active_crises:
                    print(f"  !! {c}")
            print(f"  Reward: {obs.reward}")

        messages.append(observation_to_message(obs))

        # ── Sliding window ──────────────────────────────────────────
        max_history = 2 + 2 * max_steps
        if len(messages) > max_history:
            messages = [messages[0]] + messages[-(max_history - 1):]

    # ================================================================
    # DUAL-LAYER EVALUATION
    # ================================================================

    # Layer 1: Programmatic Score
    grader_report = env.grader.report()
    programmatic_score = grader_report["verification_score"]

    # Layer 2: LLM Judge Score
    judge_report = await run_llm_judge(llm, env)
    llm_judge_score = judge_report.get("overall", 0.5)

    # Combined: 60% programmatic + 40% strategic
    final_score = round(0.6 * programmatic_score + 0.4 * llm_judge_score, 4)

    # ── Final Report ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("       DUAL-LAYER EVALUATION REPORT")
        print("=" * 60)

        print(f"\n  Steps completed: {len(observations) - 1}")
        print(f"  Budget remaining: ${env.state.budget:,.2f}")
        print(
            f"  Deliveries: {env._deliveries_on_time}/{env._deliveries_total} on-time"
        )
        print(f"  Late hours: {env._cumulative_late_hours}")

        print(f"\n{'─'*60}")
        print("  PROGRAMMATIC GRADER (Verification Score)")
        print(f"{'─'*60}")
        print(f"    Safety:      {grader_report['safety']:.4f}")
        print(f"    Reliability: {grader_report['reliability']:.4f}")
        print(f"    Efficiency:  {grader_report['efficiency']:.4f}")
        print(f"    ─────────────────────────────────")
        print(f"    Verification Score: {programmatic_score:.4f}")

        print(f"\n{'─'*60}")
        print("  LLM JUDGE (Strategic Score)")
        print(f"{'─'*60}")
        print(f"    Crisis Awareness:       {judge_report.get('crisis_awareness', 'N/A')}")
        print(f"    Strategic Redeployment: {judge_report.get('strategic_redeployment', 'N/A')}")
        print(f"    Cost Discipline:        {judge_report.get('cost_discipline', 'N/A')}")
        print(f"    ─────────────────────────────────")
        print(f"    Strategic Score: {llm_judge_score:.4f}")
        if "reasoning" in judge_report:
            print(f"    Reasoning: {judge_report['reasoning']}")

        print(f"\n{'═'*60}")
        print(f"  FINAL WEIGHTED SUBMISSION SCORE")
        print(f"  = 0.6 × {programmatic_score:.4f} + 0.4 × {llm_judge_score:.4f}")
        print(f"  = {final_score:.4f}")
        print(f"{'═'*60}")

    return {
        "observations": observations,
        "programmatic_score": programmatic_score,
        "llm_judge_score": llm_judge_score,
        "final_score": final_score,
        "grader_report": grader_report,
        "judge_report": judge_report,
    }


# ── CLI entry-point ─────────────────────────────────────────────────


def main() -> None:
    """Run the agent from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
