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

        # Field normalization for DDxPlus variants (uppercase or lowercase schema)
        self.chief_complaint = (
            case.get("initial_evidence")
            or case.get("INITIAL_EVIDENCE")
            or "Unknown complaint"
        )

        raw_evidences = case.get("evidences") or case.get("EVIDENCES") or {}

        # EVIDENCES may come as a string representation of a list in some dataset versions
        if isinstance(raw_evidences, str):
            try:
                import ast

                parsed = ast.literal_eval(raw_evidences)
                if isinstance(parsed, list):
                    raw_evidences = parsed
            except Exception:
                raw_evidences = []

        if isinstance(raw_evidences, list):
            # Convert list form to dict with implicit present flags
            self.evidences = {str(item): True for item in raw_evidences}
        elif isinstance(raw_evidences, dict):
            self.evidences = raw_evidences
        else:
            self.evidences = {}

        logger.debug(
            "PatientAgent initialized. Pathology: %s",
            case.get("PATHOLOGY", case.get("pathology", "unknown")),
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
        return {
            "chief_complaint": str(self.chief_complaint),
            "vitals": {},  # DDxPlus does not include traditional vitals
            "profile": {
                "age": self.case.get("age") or self.case.get("AGE"),
                "sex": self.case.get("sex") or self.case.get("SEX"),
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
