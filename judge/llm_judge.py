"""
llm_judge.py
------------
Handles OpenRouter API interaction utilizing Llama-3-70b-instruct to evaluate 
clinical bounds. Contains defensive JSON parsing fallbacks explicitly trapping crashes.
"""

import os
import json
import logging
import requests
from typing import Dict, Any

from judge.prompts import JUDGE_PROMPT

logger = logging.getLogger(__name__)


class LLMJudge:
    def __init__(self, model_id: str = "meta-llama/llama-3-70b-instruct"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model_id = model_id

    def evaluate(self, trajectory: list, ground_truth: str, agent_diagnosis: str) -> Dict[str, float]:
        """
        Calculates subjective episodic values mapped cleanly to the 5 dimensions required.
        """
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found. Skipping LLM judge.")
            return {
                "reasoning": 0.0,
                "information": 0.0,
                "treatment": 0.0,
                "followup": 0.0,
                "efficiency": 0.0
            }

        prompt = JUDGE_PROMPT.format(
            ground_truth=ground_truth,
            agent_diagnosis=agent_diagnosis,
            trajectory=json.dumps(trajectory, indent=2)
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a specialized medical AI reporting pure JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20.0
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            # Defensive clean stripping just in case the LLM outputs markdown braces
            content = content.replace("```json", "").replace("```", "").strip()
            
            scores = json.loads(content)
            
            # Map values dynamically validating boundaries
            required_keys = ["reasoning", "information", "treatment", "followup", "efficiency"]
            validated_scores = {}
            
            for key in required_keys:
                try:
                    val = float(scores.get(key, 0.0))
                    validated_scores[key] = max(0.0, min(10.0, val)) # Clamp strictly 0-10
                except (ValueError, TypeError):
                    validated_scores[key] = 0.0
                    
            return validated_scores

        except Exception as e:
            logger.error("LLM Judge fallback activated: API completely failed - %s", str(e))
            return {
                "reasoning": 0.0,
                "information": 0.0,
                "treatment": 0.0,
                "followup": 0.0,
                "efficiency": 0.0
            }
