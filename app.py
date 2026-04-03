import os
import json
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any
import logging
from pprint import pformat

from env.medenv import MedEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── Lazy env singleton ────────────────────────────────────────────────
_env: MedEnv | None = None

def get_env() -> MedEnv:
    global _env
    if _env is None:
        _env = MedEnv()
    return _env

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="MedAgent-Env", version="1.0.0")

# ── Request schemas ───────────────────────────────────────────────────
class ResetRequest(BaseModel):
    seed: int | None = None

class StepRequest(BaseModel):
    action_type: str
    payload: dict[str, Any] = {}

# ── REST API endpoints ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset_env(request: ResetRequest | None = None):
    seed = request.seed if request else None
    obs = get_env().reset(seed=seed)
    return dict(obs)

@app.post("/step")
def step_env(request: StepRequest):
    action = {
        "action_type": request.action_type,
        "payload": request.payload
    }
    obs, reward, terminated, truncated, info = get_env().step(action)
    return {
        "observation": obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }

@app.get("/state")
def state_env():
    return get_env().state()

# ── Gradio UI ─────────────────────────────────────────────────────────
def run_demo_episode(seed: int = 42) -> str:
    """Run a deterministic MedAgent episode and return a text log for UI."""
    env = MedEnv(seed=seed)
    output_lines = []

    output_lines.append("[1] Resetting environment")
    obs = env.reset(seed=seed)
    output_lines.append(f"Stage={obs.get('stage')} | Chief complaint={obs.get('chief_complaint')}")

    actions = [
        {"action_type": "flag_emergency", "payload": {"flag": False, "reason": "No immediate life threat"}},
        {"action_type": "ask_symptom", "payload": {"symptom": "fever"}},
        {"action_type": "order_test", "payload": {"test": "blood_panel"}},
        {"action_type": "give_diagnosis", "payload": {"diagnosis": "Viral Infection", "confidence": 0.85}},
        {"action_type": "prescribe", "payload": {"drug": "Ibuprofen", "dose": "400mg", "duration": "5 days"}},
        {"action_type": "schedule_followup", "payload": {"days": 7}},
    ]

    for i, action in enumerate(actions, start=2):
        obs, reward, terminated, truncated, info = env.step(action)
        output_lines.append(
            f"[{i}] action={action['action_type']} | stage={env.current_stage} | reward={reward:.2f} | terminated={terminated} | msg={obs.get('message')}"
        )

        if terminated or truncated:
            output_lines.append(f"Episode terminated at step {i}.")
            break

    output_lines.append("--- Finale ---")
    output_lines.append(f"Final stage: {env.current_stage}")
    output_lines.append(f"Total steps: {env.state().get('episode_step_count')}")
    output_lines.append(f"Final reward (info): {info.get('final_reward')}")
    output_lines.append("Trajectory:")
    output_lines.append(pformat(env.state().get('actions_taken')))

    return "\n".join(output_lines)


def get_state() -> str:
    env = MedEnv(seed=42)
    env.reset(seed=42)
    return pformat(env.state())


def build_gradio_ui() -> gr.Blocks:
    with gr.Blocks(title="MedAgent-Env (HuggingFace Spaces)") as demo:
        gr.Markdown("## MedAgent-Env sample run\nA deterministic clinical pipeline episode, ready for HuggingFace Spaces.")

        output_text = gr.Textbox(lines=20, label="Episode Log")

        run_button = gr.Button("Run deterministic episode")
        run_button.click(fn=run_demo_episode, inputs=[], outputs=[output_text])

        state_button = gr.Button("Show base environment state")
        state_button.click(fn=get_state, inputs=[], outputs=[output_text])

        gr.Markdown("---\n`requirements.txt` should include gradio, plus the existing runtime deps.")
    return demo

# Mount Gradio at /ui — does NOT interfere with /reset or /step
gradio_app = build_gradio_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
