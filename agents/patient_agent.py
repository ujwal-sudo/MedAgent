"""
patient_agent.py
----------------
PatientAgent — simulates the patient side of the clinical encounter.

Initialized with a raw case dict from DDxPlusLoader.

DDxPlus field mapping:
    initial_evidence  → chief_complaint
    evidences         → symptom lookup dict
    age, sex          → patient profile
    pathology         → ground-truth label (NOT exposed to GP-Agent)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PatientAgent:
    """
    Simulates a patient from a DDxPlus case row.

    Public API (Day 1):
        get_initial_info()         → chief_complaint, vitals, profile
        respond_symptom(symptom)   → stub

    Design rules:
        - All field access via .get() — dataset may have missing fields
        - PATHOLOGY is never exposed (ground-truth label leakage prevention)
    """

    def __init__(self, case: dict[str, Any]) -> None:
        if not isinstance(case, dict):
            raise TypeError(f"case must be a dict, got {type(case).__name__}")
        self.case = case
        # evidences may be a dict {symptom_id: value} or list — handle both
        raw_evidences = case.get("evidences", {})
        self.evidences: dict[str, Any] = (
            raw_evidences if isinstance(raw_evidences, dict) else {}
        )
        logger.debug(
            "PatientAgent initialized. Pathology: %s", case.get("pathology", "unknown")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_initial_info(self) -> dict[str, Any]:
        """
        Return observable initial patient information.

        Returns:
            {
                "chief_complaint": str,  # from initial_evidence field
                "vitals": dict,          # empty — not in DDxPlus
                "profile": {
                    "age": int | None,
                    "sex": str | None,
                }
            }
        """
        chief_complaint = self.case.get("initial_evidence") or "Unknown complaint"
        return {
            "chief_complaint": str(chief_complaint),
            "vitals": {},  # DDxPlus does not include traditional vitals
            "profile": {
                "age": self.case.get("age"),
                "sex": self.case.get("sex"),
            },
        }

    def respond_symptom(self, symptom_id: str) -> str:
        """
        [Day 2] Answer whether a specific symptom or test result is present.

        Args:
            symptom_id: Key from the evidences.

        Returns:
            "present", "absent", or "unknown".
        """
        # evidences can be a list or a dict in HuggingFace DDxPlus depending on how it's mapped
        if isinstance(self.evidences, list):
            if symptom_id in self.evidences:
                return "present"
            return "unknown"

        elif isinstance(self.evidences, dict):
            if symptom_id in self.evidences:
                val = self.evidences[symptom_id]
                # Try to resolve negative/positive explicitly, fallback to present
                if isinstance(val, bool):
                    return "present" if val else "absent"
                if str(val).lower() in ("false", "0", "no", "absent"):
                    return "absent"
                return "present"
            return "unknown"

        return "unknown"

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        age = self.case.get("age", "?")
        sex = self.case.get("sex", "?")
        pathology = self.case.get("pathology", "unknown")
        return f"<PatientAgent age={age} sex={sex} pathology={pathology}>"
