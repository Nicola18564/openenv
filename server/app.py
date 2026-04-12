"""OpenEnv-native FastAPI application for the placement-readiness environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

from openenv.core.env_server.http_server import create_app

from server.environment import PlacementIntelligenceEnvironment
from server.models import PlacementAction, PlacementObservation


app = create_app(
    PlacementIntelligenceEnvironment,
    PlacementAction,
    PlacementObservation,
    env_name="placement-intelligence-environment",
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
