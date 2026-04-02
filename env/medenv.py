"""
medenv.py
---------
MedEnv — Gymnasium-style clinical RL environment for the
MedAgent-Env multi-agent clinical decision-making pipeline.

Supports: reset(), step(), state() — full OpenEnv API.
"""

import logging
from typing import Any

from agents.patient_agent import PatientAgent
from data.ddxplus_loader import DDxPlusLoader
from data.ncd_sampler import NCDSampler
from agents.specialist_agent import SpecialistAgent
from env.action_handler import ActionHandler
from reward.reward_engine import RewardEngine
from schemas.observation_schema import ObservationDict

logger = logging.getLogger(__name__)

_TASK_DESCRIPTION = "Assess patient and proceed through clinical pipeline"
_TRIAGE_ACTIONS = ["flag_emergency"]


class MedEnv:
    """
    MedEnv — multi-agent clinical RL environment.

    Agents:
        PatientAgent : Initialized from a real DDxPlus case on each reset().
        GPAgent      : External agent; interacts via step() (Day 2).

    Usage:
        env = MedEnv()
        obs = env.reset(seed=42)
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:
            seed: Global seed. Can be overridden per-episode in reset().
        """
        loader = DDxPlusLoader()
        self._sampler = NCDSampler(loader, seed=seed)

        # Episode-level state
        self._patient: PatientAgent | None = None
        self._stage: str = ""
        self._step_count: int = 0
        self._current_case: dict[str, Any] | None = None

        # Day 2 state
        self._symptoms_queried: list[str] = []
        self._tests_ordered: list[str] = []
        self._test_budget_remaining: int = 5
        self._diagnosis: str | None = None
        self._diagnosis_confidence: float | None = None
        self._treatment_given: dict | None = None
        self._followup_days: int | None = None
        self._flag_emergency_used: bool = False
        self._specialist_activated: bool = False
        self._specialist_diagnosis: str | None = None
        self._specialist_treatment: dict | None = None
        self._trajectory: list[dict[str, Any]] = []
        
        self._action_handler = ActionHandler()
        self._reward_engine = RewardEngine()

        logger.info(
            "MedEnv initialized. Dataset size: %d cases.", self._sampler.dataset_size
        )

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ObservationDict:
        """
        Start a new episode.

        1. Re-seeds sampler if seed provided.
        2. Samples a patient case via NCDSampler → DDxPlusLoader.
        3. Initializes PatientAgent from the case.
        4. Resets internal state (stage="triage", step_count=0).
        5. Returns a structured ObservationDict.

        Args:
            seed    : Episode-level seed for reproducibility.
            options : Reserved for future use.

        Returns:
            ObservationDict with 7 required keys.
        """
        if seed is not None:
            self._sampler.reseed(seed)
            logger.debug("Sampler re-seeded: seed=%d", seed)

        # Sample case
        case = self._sampler.sample()
        self._current_case = case
        logger.debug("Sampled case. Pathology: %s", case.get("pathology", "unknown"))

        # Initialize patient
        self._patient = PatientAgent(case)
        info = self._patient.get_initial_info()

        # Reset state
        self._stage = "triage"
        self._step_count = 0
        self._symptoms_queried = []
        self._tests_ordered = []
        self._test_budget_remaining = 5
        self._diagnosis = None
        self._diagnosis_confidence = None
        self._treatment_given = None  # dict with drug, dose, duration
        self._followup_days = None
        self._flag_emergency_used = False
        self._specialist_activated = False
        self._specialist_diagnosis = None
        self._specialist_treatment = None
        self._trajectory = []

        chief_complaint = info["chief_complaint"]

        obs = ObservationDict(
            stage=self._stage,
            chief_complaint=chief_complaint,
            vitals=info["vitals"],
            patient_profile=info["profile"],
            available_actions=list(_TRIAGE_ACTIONS),
            task_description=_TASK_DESCRIPTION,
            message=f"Patient presents with: {chief_complaint}",
        )

        logger.info("Episode reset. %s", self._patient)
        return obs

    def step(
        self, action: dict[str, Any]
    ) -> tuple[ObservationDict, float, bool, bool, dict[str, Any]]:
        """
        Execute one GP-Agent action.
        """
        self._assert_initialized("step")
        self._step_count += 1

        message, reward, terminated = self._action_handler.handle_action(self, action)

        # Build fresh observation after action
        info_dict = self._patient.get_initial_info()
        obs = ObservationDict(
            stage=self._stage,
            chief_complaint=info_dict["chief_complaint"],
            vitals=info_dict["vitals"],
            patient_profile=info_dict["profile"],
            available_actions=self._action_handler._get_allowed_actions(self._stage),
            task_description=_TASK_DESCRIPTION,
            message=message,
        )

        # Track trajectory
        self._trajectory.append({
            "step": self._step_count,
            "action": action.copy(),
            "observation": obs["message"]
        })

        # In Day 4/5, intermediate reward is always 0. Final computed iteratively via Engine.
        final_reward = 0.0
        prog_reward = 0.0
        breakdown = {}
        llm_status = {}

        final_diagnosis = self._diagnosis or self._specialist_diagnosis
        
        # Package state safely for reward calc
        state_dict = {
            "steps_taken": self._step_count,
            "tests_ordered": self._tests_ordered.copy(),
            "diagnosis": self._diagnosis,
            "final_diagnosis": final_diagnosis,
            "treatment_given": self._treatment_given,
            "flag_emergency_used": self._flag_emergency_used,
            "specialist_activated": self._specialist_activated,
            "specialist_diagnosis": self._specialist_diagnosis,
            "specialist_treatment": self._specialist_treatment
        }

        if terminated:
            raw_case = self._current_case or {}
            prog_reward, breakdown = self._reward_engine.compute_total_reward(state_dict, raw_case)
            
            # --- Day 5 Integration: LLM Evaluation ---
            from judge.llm_judge import LLMJudge
            judge = LLMJudge()
            
            pathology = raw_case.get("PATHOLOGY", raw_case.get("pathology", "unknown"))
            agent_diag = str(final_diagnosis) if final_diagnosis else "None"
            
            llm_scores = judge.evaluate(self._trajectory, pathology, agent_diag)
            
            total_llm_score = sum(llm_scores.values()) / 50.0 * 100.0 if llm_scores else 0.0
            
            # --- Bonus Fix: Safety Override Dominance ---
            if breakdown.get("override", False):
                final_reward = -100.0
            else:
                final_reward = 0.6 * prog_reward + 0.4 * total_llm_score
                
            llm_status = llm_scores
                
            breakdown["llm_score_total"] = total_llm_score

        # Truncated always false standard setup
        truncated = False
        
        # Determine pathology defensively since structure changes per load/file
        pathology = "unknown"
        if self._current_case:
            pathology = self._current_case.get("PATHOLOGY", self._current_case.get("pathology", "unknown"))

        info = {
            "ground_truth": pathology,
            "final_diagnosis": final_diagnosis,
            "steps_taken": self._step_count,
            "diagnosis": self._diagnosis,
            "tests_used": self._tests_ordered.copy(),
            "specialist": {
                "used": self._specialist_activated,
                "diagnosis": self._specialist_diagnosis
            },
            "reward_breakdown": breakdown,
            "llm_scores": llm_status if terminated else {},
            "programmatic_reward": prog_reward if terminated else 0.0,
            "final_reward": final_reward,
        }

        return obs, float(final_reward), terminated, truncated, info

    def state(self) -> dict[str, Any]:
        """Return full serializable internal state for logging/debugging."""
        self._assert_initialized("state")
        return {
            "current_stage": self._stage,
            "episode_step_count": self._step_count,
            "symptoms_queried": self._symptoms_queried.copy(),
            "tests_ordered": self._tests_ordered.copy(),
            "test_budget_remaining": self._test_budget_remaining,
            "current_diagnosis": self._diagnosis,
            "diagnosis_confidence": self._diagnosis_confidence,
            "treatment_given": self._treatment_given,
            "followup_days": self._followup_days,
            "red_flag_status": self._flag_emergency_used,
            "specialist_activated": self._specialist_activated,
            "specialist_diagnosis": self._specialist_diagnosis,
            "specialist_treatment": self._specialist_treatment,
            "actions_taken": self._trajectory.copy(),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_case(self) -> dict[str, Any] | None:
        """Raw sampled case dict (useful for debugging in main.py)."""
        return self._current_case

    @property
    def current_stage(self) -> str:
        return self._stage

    @property
    def patient(self) -> PatientAgent | None:
        return self._patient

    @property
    def test_budget_remaining(self) -> int:
        return self._test_budget_remaining

    # ------------------------------------------------------------------
    # Mutators for ActionHandler
    # ------------------------------------------------------------------

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def record_symptom_queried(self, symptom: str) -> None:
        self._symptoms_queried.append(symptom)

    def decrement_test_budget(self) -> None:
        self._test_budget_remaining -= 1

    def record_test_ordered(self, test: str) -> None:
        self._tests_ordered.append(test)

    def record_diagnosis(self, diagnosis: str) -> None:
        self._diagnosis = diagnosis

    def set_diagnosis_confidence(self, conf: float) -> None:
        self._diagnosis_confidence = conf

    def record_treatment(self, drug: str, dose: str = "unknown", duration: str = "unspecified") -> None:
        self._treatment_given = {"drug": drug, "dose": dose, "duration": duration}

    def record_followup(self, days: int) -> None:
        self._followup_days = days

    def record_emergency_flagged(self) -> None:
        self._flag_emergency_used = True
        
    def activate_specialist(self, diagnosis: str, treatment: dict) -> None:
        self._specialist_activated = True
        self._specialist_diagnosis = diagnosis
        self._specialist_treatment = treatment

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self, method: str) -> None:
        if self._patient is None:
            raise RuntimeError(
                f"MedEnv.{method}() called before reset(). Call env.reset() first."
            )

    def __repr__(self) -> str:
        return f"<MedEnv stage={self._stage!r} step={self._step_count}>"
