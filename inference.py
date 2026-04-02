import os
import json
import requests
from openai import OpenAI

# ── Environment variables (exact names, exact defaults) ──────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/llama-3.3-70b-instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── OpenAI client (all LLM calls go through this) ────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "none"
)

# ── Environment Space URL ─────────────────────────────────────────────
# Replace with your actual deployed Space URL
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://Kn1ghtofzer0-medagent-env.hf.space"
)

def get_action_from_llm(observation: str, stage: str) -> dict:
    """Call LLM to decide next clinical action given current observation."""
    
    system_prompt = """You are a clinical decision-making AI agent 
operating inside MedAgent-Env, a reinforcement learning environment.
You must navigate a patient through triage, diagnosis, treatment,
and follow-up.

SAFETY RULE: If you detect any red flag symptoms (chest pain, severe
shortness of breath, unconsciousness, severe bleeding), you MUST
call flag_emergency immediately.

Available actions by stage:
- triage: classify_urgency, flag_emergency
- diagnosis: ask_symptom, order_test, give_diagnosis, refer, flag_emergency  
- treatment: prescribe, order_test, flag_emergency
- follow_up: schedule_followup, flag_emergency

Respond with ONLY a valid JSON action object. No explanation.
Examples:
{"action_type": "classify_urgency", "payload": {"urgency": "routine"}}
{"action_type": "ask_symptom", "payload": {"symptom_id": "chest_pain"}}
{"action_type": "order_test", "payload": {"test_id": "blood_glucose"}}
{"action_type": "give_diagnosis", "payload": {"icd_code": "E11", "confidence": 0.85}}
{"action_type": "refer", "payload": {"specialty": "endocrinology", "reason": "complex diabetes management"}}
{"action_type": "prescribe", "payload": {"drug_id": "metformin", "dose": "500mg", "duration": "90 days"}}
{"action_type": "schedule_followup", "payload": {"days": 90}}
{"action_type": "flag_emergency", "payload": {"reason": "chest pain with diaphoresis"}}
"""

    user_message = f"Current stage: {stage}\n\nObservation:\n{observation}\n\nWhat is your next clinical action?"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
        max_tokens=200
    )
    
    raw = response.choices[0].message.content.strip()
    
    # Parse JSON — handle markdown code blocks if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw.strip())


def run_episode(env_url: str) -> float:
    """Run one complete episode against the deployed environment."""
    
    print("START")
    
    # Reset environment
    reset_response = requests.post(f"{env_url}/reset", timeout=120)
    reset_response.raise_for_status()
    obs_dict = reset_response.json()
    
    observation = obs_dict.get("message", str(obs_dict))
    stage = obs_dict.get("stage", "triage")
    
    print(f"STEP 0 | stage={stage} | obs={observation[:100]}")
    
    total_reward = 0.0
    step_count = 0
    max_steps = 20
    
    while step_count < max_steps:
        # Get action from LLM
        try:
            action = get_action_from_llm(observation, stage)
        except Exception as e:
            print(f"STEP {step_count + 1} | ERROR parsing LLM action: {e}")
            # Fallback action based on stage
            fallback_map = {
                "triage":    {"action_type": "classify_urgency",
                              "payload": {"urgency": "routine"}},
                "diagnosis": {"action_type": "give_diagnosis",
                              "payload": {"icd_code": "R69",
                                          "confidence": 0.5}},
                "treatment": {"action_type": "prescribe",
                              "payload": {"drug_id": "supportive_care",
                                          "dose": "standard",
                                          "duration": "7 days"}},
                "follow_up": {"action_type": "schedule_followup",
                              "payload": {"days": 30}}
            }
            action = fallback_map.get(
                stage,
                {"action_type": "schedule_followup",
                 "payload": {"days": 30}}
            )
        
        # Execute action in environment
        step_response = requests.post(
            f"{env_url}/step",
            json=action,
            timeout=120
        )
        step_response.raise_for_status()
        result = step_response.json()
        
        reward      = result.get("reward", 0.0)
        terminated  = result.get("terminated", False)
        truncated   = result.get("truncated", False)
        obs_raw     = result.get("observation", {})
        
        # Handle observation (may be dict or string)
        if isinstance(obs_raw, dict):
            observation = obs_raw.get("message", str(obs_raw))
            stage = obs_raw.get("stage", stage)
        else:
            observation = str(obs_raw)
        
        total_reward += reward
        step_count += 1
        
        print(
            f"STEP {step_count} | "
            f"action={action.get('action_type')} | "
            f"reward={reward:.3f} | "
            f"stage={stage} | "
            f"terminated={terminated}"
        )
        
        if terminated or truncated:
            break
    
    print(f"END | total_reward={total_reward:.3f} | steps={step_count}")
    return total_reward


def from_docker_image(image_name: str) -> str:
    """Launch local Docker container for testing and return local URL."""
    import subprocess
    cmd = f"docker run -p 7860:7860 {image_name}"
    subprocess.Popen(cmd.split())
    return "http://127.0.0.1:7860"


if __name__ == "__main__":
    env_url = ENV_BASE_URL
    if LOCAL_IMAGE_NAME:
        env_url = from_docker_image(LOCAL_IMAGE_NAME)
    run_episode(env_url)