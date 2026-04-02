"""
specialist_agent.py
-------------------
SpecialistAgent — activated via the 'refer' action to provide
expert-level diagnosis and treatment recommendations.
"""


class SpecialistAgent:
    def __init__(self, case, specialty: str = "general"):
        self.case = case
        self.specialty = specialty

    def provide_diagnosis(self):
        # use ground truth as specialist output
        return self.case.get("PATHOLOGY", self.case.get("pathology", "unknown"))

    def recommend_treatment(self):
        d = self.provide_diagnosis().lower()

        if "diabetes" in d:
            return {"drug": "metformin", "dose": "500mg"}
        elif "hypertension" in d:
            return {"drug": "amlodipine", "dose": "5mg"}
        elif "urti" in d or "respiratory" in d:
            return {"drug": "paracetamol", "dose": "500mg"}
        else:
            return {"drug": "supportive_care", "dose": "standard"}
