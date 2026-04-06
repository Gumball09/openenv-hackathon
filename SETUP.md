# Setup Guide — Logistics Crisis Manager

This guide walks you through everything you need to do to get the
**Logistics Crisis Manager** OpenEnv environment running from a fresh
clone of just the `logistics_crisis_manager/` folder.

---

## 1. Prerequisites

Install these on your machine first:

- **Python 3.10+** (with `pip` and `venv`)
- **Docker** — make sure the daemon is running (`docker info` should succeed)
- **Git** (for cloning)
- An **OpenAI-compatible API key**. The baseline defaults to the
  HuggingFace router, but any OpenAI-compatible endpoint works
  (OpenAI, HF Router, vLLM, Ollama, etc.)

---

## 2. Folder layout (important)

`inference.py` imports from `logistics_crisis_manager.models`, so the
cloned directory **must be importable as a Python package named
`logistics_crisis_manager`**. After cloning, make sure the folder is
literally named `logistics_crisis_manager/` and that you will run
commands from its **parent** directory:

```bash
mkdir openenv-hackathon && cd openenv-hackathon
git clone <repo-url> logistics_crisis_manager
# or: move the cloned folder here and rename if necessary
```

Final layout should look like:

```
openenv-hackathon/
└── logistics_crisis_manager/
    ├── inference.py
    ├── models.py
    ├── server/
    ├── client/
    ├── Dockerfile
    └── ...
```

---

## 3. Create a virtualenv and install dependencies

From inside `logistics_crisis_manager/`:

```bash
cd logistics_crisis_manager
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

pip install -e .                   # installs via pyproject.toml
# or, equivalently:
# pip install -r requirements.txt
```

This installs `fastapi`, `uvicorn`, `pydantic`, `networkx`,
`openenv-core`, `openai`, and `python-dotenv`.

---

## 4. Build the Docker image

From inside `logistics_crisis_manager/`:

```bash
# Option A — using the OpenEnv CLI (preferred, reads openenv.yaml)
openenv build

# Option B — plain docker
docker build -t openenv-logistics_crisis_manager .
```

The base image is `ghcr.io/meta-pytorch/openenv-base:latest`. The
first build will pull it, which may take a few minutes.

> **Apple Silicon:** if the base image complains on M-series Macs, add
> `--platform linux/amd64` to the `docker build` command.

---

## 5. Run the environment server

```bash
docker run --rm -p 8000:8000 openenv-logistics_crisis_manager
```

Verify it's up:

```bash
curl http://localhost:8000/health
```

The FastAPI server exposes `POST /reset`, `POST /step`, `GET /state`,
`GET /schema`, and a WebSocket session endpoint at `/ws`.

---

## 6. Configure LLM API environment variables

`inference.py` reads **three required** environment variables. The
variable is named `HF_TOKEN` for historical reasons — it's just the
API key for whatever OpenAI-compatible endpoint you point at.

| Variable        | Purpose                                                 |
| --------------- | ------------------------------------------------------- |
| `API_BASE_URL`  | LLM endpoint base URL                                   |
| `MODEL_NAME`    | Model identifier                                        |
| `HF_TOKEN`      | API key for the endpoint                                |

### Using HuggingFace Router (default)

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

### Using OpenAI directly

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-xxxxxxxxxxxx        # yes, still this variable name
```

You can also place these in a `.env` file in the parent directory;
`python-dotenv` will pick them up.

### Optional overrides

| Variable               | Default | Purpose                            |
| ---------------------- | ------- | ---------------------------------- |
| `MAX_STEPS_PER_TASK`   | `20`    | Max actions per task               |
| `TEMPERATURE`          | `0.2`   | Sampling temperature               |
| `MAX_OUTPUT_TOKENS`    | `350`   | Max tokens per LLM response        |

---

## 7. Run the baseline

`inference.py` adds its parent directory to `sys.path`, so run it from
the **parent** of `logistics_crisis_manager/`:

```bash
cd ..                               # now in the parent folder
python logistics_crisis_manager/inference.py
```

You should see structured log lines for all three tiered tasks:

```
[START]   {"task_id":"easy",...}
[STEP]    {"task_id":"easy","step":1,...}
[END]     {"task_id":"easy","score":0.75,...}
[SUMMARY] {"scores":{"easy":0.75,"medium":0.66,"hard":0.45},...}
```

---

## 8. (Optional) Streamlit dashboard

An interactive UI for resetting tiers, sending actions, and watching
grader sub-scores live:

```bash
pip install streamlit pandas
streamlit run logistics_crisis_manager/client/dashboard.py
```

Make sure the Docker server from step 5 is running on
`http://localhost:8000`.

---

## Troubleshooting

- **`ModuleNotFoundError: logistics_crisis_manager`** — you ran
  `inference.py` from the wrong directory, or the folder isn't named
  exactly `logistics_crisis_manager`. Run from the parent directory.
- **`Connection refused` on port 8000** — the Docker container isn't
  running, or the port is already in use. Try
  `docker run -p 8001:8000 ...` and update your client accordingly.
- **`openenv: command not found`** — `pip install -e .` installs
  `openenv-core` which provides the CLI. Re-activate your venv, or
  skip it and use `docker build` directly.
- **401 / 403 from the LLM** — `HF_TOKEN` isn't exported in the same
  shell where you run `inference.py`, or the key is invalid for the
  endpoint you chose.
- **Docker base image errors on Apple Silicon** — add
  `--platform linux/amd64` to `docker build` and `docker run`.
