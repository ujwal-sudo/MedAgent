"""
safety.py
---------
Checks if the case contains critical evidences (red flags).
Override rule: If a red flag is missed and the agent progresses without flagging an emergency, penalty = -100.
"""

from typing import Any, Tuple

# Baseline definitions. Extending these arrays provides broader real-world coverage.
CRITICAL_EVIDENCES = {"chest_pain", "severe_shortness_of_breath", "unconsciousness", "E_129", "E_45"}


def compute_safety_reward(state: dict[str, Any], case: dict[str, Any]) -> Tuple[float, bool]:
    """
    Evaluates safety conditions.
    
    Returns:
        reward: The float scalar output.
        override: Bool. If True, terminates remaining RewardEngine logic.
    """
    has_red_flag = False
    
    # HuggingFace datasets often list strings or dictionaries for valid evidences depending on the format logic.
    raw_evidences = case.get("EVIDENCES", case.get("evidences", []))
    evidences = list(raw_evidences.keys()) if isinstance(raw_evidences, dict) else raw_evidences
    
    for flag in CRITICAL_EVIDENCES:
        if flag in evidences:
            has_red_flag = True
            break
            
    flagged = state.get("flag_emergency_used", False)
    
    if has_red_flag and not flagged:
        return -100.0, True  # Extreme penalty dictating Safety Override
    
    if has_red_flag and flagged:
        return 30.0, False   # Commended for successfully catching danger

    if not has_red_flag:
        return 10.0, False   # Standard safe baseline
        
    return 0.0, False
