"""
observation_schema.py
---------------------
Defines the structured observation returned by MedEnv.reset() and MedEnv.step().
Using TypedDict keeps it lightweight while giving type-checker clarity.
"""

from typing import Any, TypedDict


class VitalsDict(TypedDict, total=False):
    """
    Patient vital signs. All fields are optional — dataset may be incomplete.
    """
    heart_rate: float          # bpm
    blood_pressure_systolic: float   # mmHg
    blood_pressure_diastolic: float  # mmHg
    temperature: float         # °C
    respiratory_rate: float    # breaths/min
    oxygen_saturation: float   # SpO2 %


class PatientProfileDict(TypedDict, total=False):
    """
    Demographic and background information. Fields are dataset-dependent.
    """
    age: int
    sex: str
    initial_evidence: list[str]
    pathology: str             # ground-truth label (not exposed to GP-Agent)


class ObservationDict(TypedDict):
    """
    Full observation returned to the GP-Agent at each step.

    Fields:
        stage             : Current pipeline stage (e.g. "triage", "workup").
        chief_complaint    : Patient's presenting complaint in natural language.
        vitals            : Measured vital signs.
        patient_profile    : Demographic / background info.
        available_actions  : List of ActionType strings the agent may call.
        task_description   : High-level instruction for the GP-Agent.
        message            : Optional contextual message from the environment.
    """
    stage: str
    chief_complaint: str
    vitals: dict[str, Any]
    patient_profile: dict[str, Any]
    available_actions: list[str]
    task_description: str
    message: str


def empty_observation() -> ObservationDict:
    """Returns a blank ObservationDict — useful for testing and placeholders."""
    return ObservationDict(
        stage="",
        chief_complaint="",
        vitals={},
        patient_profile={},
        available_actions=[],
        task_description="",
        message="",
    )
