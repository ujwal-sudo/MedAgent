"""
reward_engine.py
----------------
Computes the final episodic reward by delegating to modular scoring functions.
Enforces the safety override logic and applies component weights.
"""

from typing import Any, Dict, Tuple

from reward.safety import compute_safety_reward
from reward.accuracy import compute_accuracy_reward
from reward.efficiency import compute_efficiency_reward
from reward.treatment import compute_treatment_reward

# Component weights — must sum to 1.0
WEIGHTS = {
    "safety": 0.40,
    "accuracy": 0.35,
    "efficiency": 0.15,
    "treatment": 0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"


class RewardEngine:
    def compute_total_reward(
        self, state: dict[str, Any], case: dict[str, Any]
    ) -> Tuple[float, dict[str, Any]]:
        """
        Calculates the weighted reward scalar and breakdown metric dictionaries.

        Safety override check happens BEFORE any weight multiplication.
        Override returns -100.0 directly — weights are NOT applied to -100.

        Formula (no override):
            total = 0.40*safety + 0.35*accuracy + 0.15*efficiency + 0.10*treatment
        """
        # Safety overriding rule — checked BEFORE weight multiplication
        safety_reward, safety_override = compute_safety_reward(state, case)

        if safety_override:
            # Return -100.0 directly — bypasses weighted sum entirely
            return -100.0, {
                "safety": safety_reward,
                "accuracy": 0.0,
                "efficiency": 0.0,
                "treatment": 0.0,
                "override": True,
            }

        # Calculate standard component scores
        accuracy_reward = compute_accuracy_reward(state, case)
        efficiency_reward = compute_efficiency_reward(state)
        treatment_reward = compute_treatment_reward(state, case)

        # Apply weights
        total = (
            WEIGHTS["safety"] * safety_reward
            + WEIGHTS["accuracy"] * accuracy_reward
            + WEIGHTS["efficiency"] * efficiency_reward
            + WEIGHTS["treatment"] * treatment_reward
        )

        breakdown = {
            "safety": safety_reward,
            "accuracy": accuracy_reward,
            "efficiency": efficiency_reward,
            "treatment": treatment_reward,
            "override": False,
        }

        return float(total), breakdown
