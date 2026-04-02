# MedAgent-Env: RL Environment for Clinical Decision-Making

> An OpenEnv-compliant reinforcement learning environment for training AI agents in end-to-end clinical decision-making.

## Problem
The clinical decision-making pipeline intrinsically requires balancing multiple objectives: prioritizing patient safety, maintaining diagnostic accuracy, and optimizing time/resource efficiency. Existing reinforcement learning (RL) environments in healthcare typically treat diagnosis purely as a classification constraint, lacking the dynamic context of clinical dialogue. Furthermore, rule-based reward curves struggle to capture the complex, semantic rationale behind a clinician's line of questioning.

## Solution
**MedAgent-Env** closes this gap by introducing a multi-agent, Gymnasium-style environment powered by large-scale synthetic patient cases from the DDxPlus dataset (NeurIPS 2022).
- **Multi-Agent System**: Simulates realistic patient flow utilizing a PatientAgent managing dynamic symptom disclosure and a SpecialistAgent for escalated clinical consults.
- **Structured Actions**: Empowers the external GP Agent to act via clean JSON/Dictionary schemas representing physical actions (`ask_symptom`, `order_test`, `flag_emergency`, etc).
- **Safety-First Reward**: An overriding constraint mechanism punishes agents that miss critical red-flag symptoms, enforcing safety as a hard constraint so RL models cannot optimize accuracy at the expense of patient survival.
- **LLM-as-Judge**: Leverages a robust evaluation pipeline to contextually score the agent's episode trajectory across subjective parameters, heavily augmenting the deterministic programmatic scalar.

## Features
* **OpenEnv/Gymnasium Compliant API**: Instantly deployable into standard RL pipelines via familiar `reset()` and `step()` loops.
* **Fully RL-Compatible**: Provides (state, action, reward) interface enabling direct integration with reinforcement learning algorithms such as PPO, DQN, and policy gradient methods.
* **DDxPlus Dataset Integration**: Environment resets natively sample synthetic patient cases from the HuggingFace `aai530-group6/ddxplus` dataset.
* **Multi-Agent Architecture**: Realistic hand-offs isolating internal variables so RL agents learn through interactive prompts (like referrals) rather than internal weights manipulation.
* **Hybrid Evaluation**: Novel 60/40 reward blending combining programmable, objective correctness with qualitative, semantic review metrics.

## Example Episode
The GP-Agent natively traverses the clinical decision-making pipeline through distinct clinical stages: `triage` → `diagnosis` → `treatment` → `follow-up`.

```text
▶ Step 1 | Action: ask_symptom
  Stage    : diagnosis
  Message  : Symptom shortness of breath: True

▶ Step 2 | Action: refer
  Stage    : treatment
  Message  : Patient referred to specialist. Specialist diagnosis: URTI

▶ Step 3 | Action: prescribe
  Stage    : follow-up
  Message  : Treatment prescribed: paracetamol

▶ Step 4 | Action: schedule_followup
  Stage    : follow-up
  Message  : Follow-up scheduled in 30 days
```

## Reward System
MedAgent-Env enforces strict medical constraints within its integrated `RewardEngine`.
* **Safety Override (Hard Constraint)**: Missing a critical red-flag symptom immediately assigns a -100 reward, overriding all other metrics. This enforces real-world clinical priority: patient safety over optimization.
* **Accuracy**: String-driven progression verifying ground-truth pathologies against agent diagnoses or successful specialist mappings (`+35`).
* **Efficiency**: Implements soft test-budget decay mechanics promoting optimal symptom inquiries vs broad test orders.
* **Treatment**: Reward modifiers tracking appropriate drug issuance (`+20`) while aggressively clamping incorrect dosage prescriptions.
* **LLM-as-Judge Evaluation**: A structured evaluation pipeline using LLMs (via OpenRouter/Llama-3) scores full episode trajectories across reasoning quality, information gathering, treatment correctness, and efficiency. These scores are normalized and blended with deterministic reward signals.

## System Overview
```text
GP Agent (External RL Model)
        ↓
     MedEnv
        ↓
  ┌───────────────┐
  │ PatientAgent  │
  │ SpecialistAgent │
  └───────────────┘
        ↓
 Reward Engine + LLM Judge
```

## How to Run
#### Setup
Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/your-org/medagent-env.git
cd medagent-env
pip install -r requirements.txt
```

#### Execute Test Pipeline
We have packaged a static Day 7 debugging array sequentially built into the runtime wrapper. Executing the runner locally will trigger the interactive loop test and emit identical environment observation dicts that a real GP-agent would traverse.

```bash
export OPENROUTER_API_KEY="sk-or-v1-YOUR_KEY" # Optional: omitting runs pure programmatic fallback
python main.py
```

## Deployment
MedAgent-Env is built directly with remote integration and model-agnostic execution in mind.

To deploy via HuggingFace Spaces:
1. Initialize a new Docker-based HuggingFace Space.
2. Link this repository securely via Git.
3. HuggingFace will utilize the provided `Dockerfile` ensuring the `datasets` library pre-allocates caching arrays dynamically.
4. Pass any model API endpoints directly via HuggingFace Secrets GUI.

The app utilizes lightweight text states mapping exactly to generic endpoints, enabling zero-GPU cold starts on cloud containers immediately.

## Dataset Credit
The core synthetic patient cases backing the RL samplers are sourced gracefully from the NeurIPS 2022 release: **DDxPlus Dataset**. We commend the original authors for developing an immense, rigorous pathology matrix suitable for robust AI tooling.
