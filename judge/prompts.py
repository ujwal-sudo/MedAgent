"""
prompts.py
----------
Houses the deterministic evaluation LLM template.
Forces JSON extraction on 5 clinical dimensions exactly mapping the 0-10 rubrics.
"""

JUDGE_PROMPT = """
You are a medical AI evaluator.

Evaluate the following clinical decision-making trajectory.

GROUND TRUTH DIAGNOSIS:
{ground_truth}

AGENT DIAGNOSIS:
{agent_diagnosis}

EPISODE TRAJECTORY:
{trajectory}

Score each category from 0 to 10:

1. Reasoning Quality
2. Information Gathering Strategy
3. Treatment Appropriateness
4. Follow-up Adequacy
5. Efficiency

Return exactly a JSON object in this format (and nothing else):
{{
  "reasoning": X,
  "information": X,
  "treatment": X,
  "followup": X,
  "efficiency": X
}}
"""
