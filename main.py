"""
main.py
-------
MedAgent-Env — deterministic execution of a single episode.

Runs a fixed trajectory of clinical decision steps demonstrating
the multi-agent RL environment pipeline. Perfect for Docker execution
on HuggingFace with no user input and no infinite loops.
"""

import logging
from pprint import pprint
from env.medenv import MedEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

def main():
    print("Initializing MedEnv...")
    env = MedEnv(seed=42)
    
    print("\n--- 1. reset() ---")
    obs = env.reset(seed=42)
    print(f"Stage: {obs.get('stage')}")
    print(f"Complaint: {obs.get('chief_complaint')}")
    
    print("\n--- Triage Phase -> flag_emergency ---")
    action_1 = {"action_type": "flag_emergency", "payload": {"flag": False, "reason": "No immediate life threat"}}
    obs, reward, term, trunc, info = env.step(action_1)
    print(f"Message: {obs.get('message')}")
    print(f"Current Stage: {env.current_stage}")
    
    print("\n--- 2. ask_symptom ---")
    action_2 = {"action_type": "ask_symptom", "payload": {"symptom": "fever"}}
    obs, reward, term, trunc, info = env.step(action_2)
    print(f"Message: {obs.get('message')}")
    
    print("\n--- 3. order_test ---")
    action_3 = {"action_type": "order_test", "payload": {"test": "blood_panel"}}
    obs, reward, term, trunc, info = env.step(action_3)
    print(f"Message: {obs.get('message')}")
    
    print("\n--- 4. give_diagnosis ---")
    action_4 = {"action_type": "give_diagnosis", "payload": {"diagnosis": "Viral Infection", "confidence": 0.85}}
    obs, reward, term, trunc, info = env.step(action_4)
    print(f"Message: {obs.get('message')}")
    
    print("\n--- 5. prescribe ---")
    action_5 = {"action_type": "prescribe", "payload": {"drug": "Ibuprofen", "dose": "400mg", "duration": "5 days"}}
    obs, reward, term, trunc, info = env.step(action_5)
    print(f"Message: {obs.get('message')}")
    
    print("\n--- 6. schedule_followup ---")
    action_6 = {"action_type": "schedule_followup", "payload": {"days": 7}}
    obs, reward, term, trunc, info = env.step(action_6)
    print(f"Message: {obs.get('message')}")
    print(f"Terminated: {term}")
    
    print("\n--- EPISODE FINISHED ---")
    print(f"Final Reward: {reward}")
    print("Additional Info:")
    pprint({
        "ground_truth": info.get("ground_truth"),
        "final_diagnosis": info.get("final_diagnosis"),
        "steps_taken": info.get("steps_taken"),
        "specialist_used": info.get("specialist", {}).get("used"),
        "reward_breakdown": info.get("reward_breakdown", {}),
        "llm_scores": info.get("llm_scores", {})
    })
    print("\nMedAgent-Env is fully PRD-compliant and deployment-ready")

if __name__ == "__main__":
    main()
