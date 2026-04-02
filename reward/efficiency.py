"""
efficiency.py
-------------
Validates exploration depth vs clinical necessities.
Discourages shotgun testing while rewarding swift precision bounds.
"""

from typing import Any


def compute_efficiency_reward(state: dict[str, Any]) -> float:
    """
    Computes an efficiency scalar clamped safely between [-10, +15].
    """
    steps = state.get("steps_taken", 0)
    tests = len(state.get("tests_ordered", []))
    
    # Base configuration starting bounds
    reward = 15.0
    
    # -1 per test 
    reward -= float(tests)
    
    # -1 per superfluous step beyond optimal thresholds
    ideal_threshold = 5
    if steps > ideal_threshold:
        reward -= float(steps - ideal_threshold)
        
    # Clamp bounding box safety
    return max(-10.0, min(15.0, reward))
