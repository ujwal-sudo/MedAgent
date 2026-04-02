"""
treatment.py
------------
Basic deterministic mapper for given diagnosis mapping strings to explicit drugs.
"""

from typing import Any

def map_treatment(diagnosis: str) -> dict[str, str]:
    d = diagnosis.lower()

    if "diabetes" in d:
        return {"drug": "metformin", "dose": "500mg"}
    elif "hypertension" in d:
        return {"drug": "amlodipine", "dose": "5mg"}
    elif "urti" in d or "respiratory" in d:
        return {"drug": "paracetamol", "dose": "500mg"}
    else:
        return {"drug": "supportive_care", "dose": "standard"}


def compute_treatment_reward(state: dict[str, Any], case: dict[str, Any]) -> float:
    """
    Evaluates validity of prescription mapping dynamically comparing the 
    underlying label rules. treatment_given is a dict with drug, dose, duration.
    """
    treatment = state.get("treatment_given")
    if not treatment:
        return 0.0  # Safe ignore rule
        
    gt = str(case.get("PATHOLOGY", case.get("pathology", ""))).lower()
    
    # Extract drug from treatment (dict or str for backward compat)
    if isinstance(treatment, dict):
        prescribed_drug = str(treatment.get("drug", "")).lower()
    else:
        prescribed_drug = str(treatment).lower()
    
    expected_mapping = map_treatment(gt)
    expected_drug = expected_mapping["drug"]
            
    if expected_drug in prescribed_drug or prescribed_drug in expected_drug:
        return 20.0
    
    return -10.0
