"""OpenEnv-native FastAPI application for the MediAssist triage environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openenv.core.env_server.http_server import create_app

from server.models import MediAssistAction, MediAssistObservation
from server.triage_environment import MediAssistTriageEnvironment


app = create_app(
    MediAssistTriageEnvironment,
    MediAssistAction,
    MediAssistObservation,
    env_name="mediassist-triage-arena",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int | None = None):
    """Run the OpenEnv server with uvicorn."""
    import uvicorn

    if port is None:
        port = int(os.getenv("PORT", os.getenv("API_PORT", "7860")))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
