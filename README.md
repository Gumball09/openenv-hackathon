# Logistics Crisis Manager — OpenEnv Environment

A reasoning-and-tool-use environment in which an LLM agent runs a 5-node
global supply chain (Shanghai · Rotterdam · London · Los Angeles · New
York) and must survive deterministic Black-Swan disruptions while
keeping high-demand hubs stocked, deliveries on time, and the budget
intact.

Built for the **Meta OpenEnv hackathon** and designed to satisfy 100% of
the OpenEnv Functional Requirements: tiered tasks, programmatic
deterministic graders, an OpenAI-compatible inference baseline, and
typed action / observation / state schemas.

---

## Motivation

Real-world logistics is one of the highest-leverage applications of
agentic AI: a single port closure, fuel-price shock, or viral demand
spike can cascade into millions of dollars of lost revenue within
hours. Crisis response requires the *exact* mix of skills modern
frontier models struggle with:

- **Long-horizon planning** under uncertainty
- **Tool use** with strict typed JSON actions
- **Resource trade-offs** (cheap-but-slow Sea vs. fast-but-expensive Air)
- **Reading dynamic context** (a streaming "News Feed" of crises)
- **Pre-positioning** stock *before* a forecasted shock fires

Simulating these crises in a deterministic, reproducible environment
gives us a high-signal benchmark that maps directly to a real
operations-research problem — and evaluates skills that pure code or
math benchmarks cannot.

---

## Action Space

Three typed actions (Pydantic models in `models.py`):

| Action            | Fields                                                                                          | Description                                              |
| ----------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `move_cargo`      | `shipment_id: str`, `route_id: str`, `carrier_type: Air \| Sea \| Rail`, `rationale: str`       | Dispatch all inventory at the route's origin.            |
| `redeploy_stock`  | `from_city: str`, `to_city: str`, `qty: int > 0`, `rationale: str`                              | Instantly transfer stock between two graph-connected cities. |
| `wait`            | `hours: int > 0`, `rationale: str`                                                              | Advance the simulation clock without acting.             |

`route_id` format: `"<Origin>-><Destination>|<Carrier>"`, e.g.
`"Shanghai->Los Angeles|Sea"`.

Every action carries a `rationale` field that the grader and LLM judge
can inspect.

## Observation Space

`LogisticsCrisisManagerObservation` (Pydantic):

| Field            | Type            | Notes                                                       |
| ---------------- | --------------- | ----------------------------------------------------------- |
| `summary`        | `str`           | Human-readable snapshot: budget, inventory per city, in-transit, recent news. |
| `active_crises`  | `list[str]`     | Currently active crisis bulletins.                          |
| `active_delays`  | `list[str]`     | Currently active delay/failure messages.                    |
| `reward`         | `float`         | Composite reward for the most recent step.                  |
| `done`           | `bool`          | Episode termination flag.                                   |
| `metadata`       | `dict`          | OpenEnv `info` channel — see below.                         |

`metadata` carries the structured `info` payload used by the grader and
training infra:

```jsonc
{
  "task_id": "hard",
  "step_count": 17,
  "current_time": 68,
  "budget_remaining": 81234.0,
  "total_spent": 18766.0,
  "deliveries_total": 12,
  "deliveries_on_time": 11,
  "on_time_rate": 0.9166,
  "in_transit": 3,
  "events": {
    "port_strike": true,
    "viral_trend": true,
    "fuel_surge": true,
    "cyber_attack": false
  }
}
```

## State

`env.state` (`LogisticsCrisisManagerState`) exposes the full underlying
simulation for debugging and replay: `nodes`, `edges`, `inventory`,
`current_time`, `budget`, `news_feed`, `episode_id`, `step_count`.

---

## Tiered Tasks

`reset(task_id=...)` selects a deterministic scenario:

### `easy` — Routine Logistics

- **Setup:** No Black-Swan events. Reduced consumer demand drain (5
  units / tick).
- **Goal:** Deliver 50 units within budget using the most cost-effective
  routes.
- **Pass criteria:** programmatic score ≥ ~0.75 with sensible Sea-only
  shipping.

### `medium` — Coastal Blockade

- **Setup:** LA Port Strike fires at **step 5** — every Sea route
  to/from Los Angeles is blocked (transit → 999 h).
- **Goal:** Maintain on-time delivery rate **> 80%** by re-routing to
  Rail / Air or via London.
- **Pass criteria:** the agent must recognise the news bulletin and
  abandon Sea routes to LA. Failing the 80% bar caps the task-progress
  sub-score by 40%.

### `hard` — Global Collapse

- **Setup:** Port Strike (step 5) + global Fuel Surge tripling all
  shipping costs (step 10) + NY Viral Trend pushing demand to 5×
  (step 15).
- **Goal:** Manage extreme NY demand spikes while fuel costs triple,
  pre-position inventory, and prioritise budget.
- **Pass criteria:** the grader hard-caps reliability at 0.2 if zero
  deliveries land in NY during the Viral Trend, dragging the final
  score below 0.5.

All three scenarios are fully deterministic — same `seed`, same
`task_id` ⇒ same trajectory. This makes the environment a structured
evaluation tool rather than a chaos simulator.

---

## Programmatic Grader

`server/grader.py` ships a `LogisticsGrader` (an OpenEnv `Rubric`) that
returns a strict score in `[0.0, 1.0]`:

```
score = 0.20 · safety
      + 0.30 · reliability
      + 0.25 · efficiency       (0.7 · OnTimeRate + 0.3 · BudgetEff)
      + 0.25 · task_progress    (deliveries / target × on-time gate)
```

- **Partial credit** is given linearly via `task_progress`: half the
  delivery target with on-time bar met ⇒ ~0.5 contribution.
- **Destructive actions** (e.g. shipping Sea to a blocked LA port) set
  `last_step_safety = 0.0` for that step and accumulate into the
  cumulative safety sub-score.
- **Hard task strictness:** an unhandled Viral Trend caps reliability
  at 0.2 *and* halves task-progress.

---

## Setup & Usage

### 1 · Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -e .          # uses pyproject.toml
```

### 2 · Build the OpenEnv container

```bash
openenv build              # builds Docker image  openenv-logistics_crisis_manager
```

### 3 · Run the server (Docker)

```bash
docker run -p 8000:8000 openenv-logistics_crisis_manager
```

The FastAPI server exposes `POST /reset`, `POST /step`, `GET /state`,
`GET /schema`, and a WebSocket session endpoint at `/ws`.

### 4 · Run the baseline inference script

The hackathon evaluator runs **`inference.py`** at the project root.
It uses the standard `openai` Python client and reads three required
environment variables:

| Variable        | Purpose                                                          |
| --------------- | ---------------------------------------------------------------- |
| `API_BASE_URL`  | LLM endpoint (e.g. `https://router.huggingface.co/v1`).          |
| `MODEL_NAME`    | Model identifier (e.g. `Qwen/Qwen2.5-72B-Instruct`).             |
| `HF_TOKEN`      | HuggingFace / OpenAI-compatible API key.                         |

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxx
python inference.py
```

`inference.py` iterates through **all three tiered tasks** and emits
structured stdout lines that the evaluator parses:

```
[START] {"task_id":"easy","model":"...","max_steps":20,...}
[STEP]  {"task_id":"easy","step":1,"action_type":"wait","action":{...},"reward":1.5,"done":false,"info":{...}}
[END]   {"task_id":"easy","score":0.7500,"safety":1.0,...}
[SUMMARY] {"scores":{"easy":0.75,"medium":0.66,"hard":0.45},"average_score":0.62,...}
```

Optional environment overrides: `MAX_STEPS_PER_TASK` (default 20),
`TEMPERATURE` (default 0.2), `MAX_OUTPUT_TOKENS` (default 350).

### 5 · Streamlit dashboard (optional UI)

```bash
pip install streamlit pandas
cd /home/shubh/openenv-hackathon
streamlit run logistics_crisis_manager/client/dashboard.py
```

The dashboard lets you reset any tier, send actions interactively,
and watch live grader sub-scores, news feed, inventory, and in-transit
shipments.

---

## Project Layout

```
logistics_crisis_manager/
├── openenv.yaml               # OpenEnv spec metadata + tiered task list
├── pyproject.toml
├── README.md                  ← this file
├── inference.py               # ★ HACKATHON ENTRY POINT (root) — uses
│                                  API_BASE_URL / MODEL_NAME / HF_TOKEN
│                                  and emits [START]/[STEP]/[END] logs
├── models.py                  # Pydantic Action / Observation / State
├── server/
│   ├── app.py                 # FastAPI mount point (create_app)
│   ├── environment.py         # LogisticsEnv — tiered reset(), step(), state
│   ├── grader.py              # LogisticsGrader (Rubric, [0,1] score)
│   └── Dockerfile
└── client/
    ├── inference.py           # legacy CLI runner (kept for local dev)
    └── dashboard.py           # Streamlit dashboard
```
