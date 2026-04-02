"""
action_schema.py
----------------
Defines the strict structured action format for GP-Agent interactions.
All actions follow the ActionDict contract — no loose dicts allowed downstream.
"""

from typing import Any, Literal, TypedDict

# All valid action types in the clinical pipeline
ActionType = Literal[
    "classify_urgency",
    "ask_symptom",
    "order_test",
    "give_diagnosis",
    "refer",
    "prescribe",
    "schedule_followup",
    "flag_emergency",
]


class ActionDict(TypedDict):
    """
    Structured action format for GP-Agent.

    Fields:
        type    : One of the predefined ActionType literals.
        payload : Arbitrary key-value data relevant to the action.
                  Schema varies by action type — validated at step time.
    """

    type: ActionType
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Convenience constructors (optional helpers for the GP-Agent to use)
# ---------------------------------------------------------------------------


def make_action(action_type: ActionType, **payload: Any) -> ActionDict:
    """
    Factory helper for building a valid ActionDict.

    Example:
        action = make_action("ask_symptom", symptom_id="S_42")
    """
    return ActionDict(type=action_type, payload=payload)


def validate_action(action: dict[str, Any]) -> ActionDict:
    """
    Lightweight validation that a raw dict is ActionDict-compatible.
    Raises ValueError on invalid structure.
    """
    if "type" not in action:
        raise ValueError("Action must contain a 'type' field.")
    valid_types = {
        "classify_urgency",
        "ask_symptom",
        "order_test",
        "give_diagnosis",
        "refer",
        "prescribe",
        "schedule_followup",
        "flag_emergency",
    }
    if action["type"] not in valid_types:
        raise ValueError(
            f"Invalid action type '{action['type']}'. "
            f"Must be one of: {sorted(valid_types)}"
        )
    if "payload" not in action or not isinstance(action["payload"], dict):
        raise ValueError("Action must contain a 'payload' dict.")
    return ActionDict(type=action["type"], payload=action["payload"])
