"""Root entry point for the OpenEnv hackathon submission.

This thin wrapper delegates to ``logistics_crisis_manager.client.inference``
so the submission spec — which expects a top-level ``client.py`` — is
satisfied without duplicating the agent logic.

Run as::

    python client.py

Honored environment variables (see ``client/inference.py`` for details):

    API_BASE_URL  default: https://api-inference.huggingface.co/v1/
    MODEL_NAME    default: Qwen/Qwen2.5-7B-Instruct
    HF_TOKEN      no default — required
"""

from __future__ import annotations

import os
import sys

# Make the parent directory importable so ``logistics_crisis_manager``
# resolves whether this script is launched from inside the package
# directory or from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from logistics_crisis_manager.client.inference import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
