"""
main.py
-------
MedAgent-Env — FastAPI server exposing the RL environment as a REST API.

Endpoints:
    POST /reset   -> calls env.reset(), returns ObservationDict
    POST /step    -> calls env.step(action), returns step result
    GET  /state   -> calls env.state(), returns full serializable state

Usage:
    python main.py
    # or: uvicorn main:app --host 0.0.0.0 --port 7860
"""

import logging
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from pydantic import BaseModel

from env.medenv import MedEnv

app = FastAPI(
    title="MedAgent-Env",
    description="OpenEnv-compliant RL environment for clinical decision-making",
    version="1.0.0",
)

# Single MedEnv instance — persists across requests
env = MedEnv()


# --- Request / Response models ---

class ResetRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    action_type: str
    payload: dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


# --- Endpoints ---

@app.post("/reset")
def reset_env(request: ResetRequest | None = None) -> dict[str, Any]:
    """Start a new episode. Optionally provide a seed for reproducibility."""
    seed = request.seed if request else None
    obs = env.reset(seed=seed)
    return dict(obs)


@app.post("/step")
def step_env(request: StepRequest) -> StepResponse:
    """Execute one GP-Agent action in the environment."""
    action = {"action_type": request.action_type, "payload": request.payload}
    obs, reward, terminated, truncated, info = env.step(action)
    return StepResponse(
        observation=dict(obs),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/state")
def get_state() -> dict[str, Any]:
    """Return the full serializable internal state of the environment."""
    return env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
