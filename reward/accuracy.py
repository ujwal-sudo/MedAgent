"""
accuracy.py
-----------
Diagnoses vs ground-truth scoring parameters using substring categorization.
"""

from typing import Any


def compute_accuracy_reward(state: dict[str, Any], case: dict[str, Any]) -> float:
    """
    Scores diagnostic precision using deterministic string comparison.
    Uses final_diagnosis (agent diagnosis || specialist diagnosis).
    """
    final_diag = state.get("final_diagnosis")
    specialist_activated = state.get("specialist_activated", False)
    spec_diag = state.get("specialist_diagnosis")
    
    gt = str(case.get("PATHOLOGY", case.get("pathology", ""))).lower().strip()
    
    # No diagnosis from any source
    if not final_diag:
        return -20.0
        
    if not gt:
        return 0.0

    pred = str(final_diag).lower().strip()
        
    # Core scoring
    if pred == gt:
        core_score = 35.0
    elif pred in gt or gt in pred:
        core_score = 20.0
    else:
        core_score = -10.0
        
    # Specialist referral modifier
    if specialist_activated:
        spec_pred = str(spec_diag).lower().strip() if spec_diag else ""
        if spec_pred == gt or spec_pred in gt or gt in spec_pred:
            core_score += 5.0  # Correct referral bonus
        else:
            core_score -= 2.0  # Unnecessary/wrong referral penalty
            
    return core_score
