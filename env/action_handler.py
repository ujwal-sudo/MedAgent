"""
action_handler.py
-----------------
Handles action validation, routing, and state transitions for MedEnv.
Enforces clinical staging rules and execution logic for all actions.
"""

from typing import Any, Tuple


class ActionHandler:
    """
    Decouples step logic from MedEnv state.
    Returns (message, reward, terminated).
    """

    def handle_action(self, env: Any, action: dict[str, Any]) -> Tuple[str, float, bool]:
        action_type = action.get("action_type")
        payload = action.get("payload", {})

        if not action_type or not isinstance(payload, dict):
            return "Invalid action structure: missing 'action_type' or 'payload'", -2.0, False

        # --- Stage Validation Rule ---
        allowed_actions = self._get_allowed_actions(env.current_stage)
        if action_type not in allowed_actions:
            return f"Action '{action_type}' not allowed in stage '{env.current_stage}'", -2.0, False

        # --- Action Routing ---
        if action_type == "classify_urgency":
            return self._handle_classify_urgency(env, payload)
        elif action_type == "ask_symptom":
            return self._handle_ask_symptom(env, payload)
        elif action_type == "order_test":
            return self._handle_order_test(env, payload)
        elif action_type == "give_diagnosis":
            return self._handle_give_diagnosis(env, payload)
        elif action_type == "refer":
            return self._handle_refer(env, payload)
        elif action_type == "prescribe":
            return self._handle_prescribe(env, payload)
        elif action_type == "schedule_followup":
            return self._handle_schedule_followup(env, payload)
        elif action_type == "flag_emergency":
            return self._handle_flag_emergency(env, payload)
        
        return f"Unhandled action type: '{action_type}'", -2.0, False

    def _get_allowed_actions(self, stage: str) -> list[str]:
        if stage == "triage":
            return ["flag_emergency", "classify_urgency"]
        elif stage == "diagnosis":
            return ["ask_symptom", "order_test", "give_diagnosis", "refer", "flag_emergency"]
        elif stage == "treatment":
            return ["prescribe", "order_test", "flag_emergency"]
        elif stage == "follow-up":
            return ["schedule_followup", "flag_emergency"]
        return []

    # ------------------------------------------------------------------
    # Specific Handlers
    # ------------------------------------------------------------------

    def _handle_classify_urgency(self, env: Any, payload: dict[str, Any]) -> Tuple[str, float, bool]:
        """Explicit triage completion — the ONLY way to advance from triage to diagnosis."""
        urgency_level = payload.get("urgency_level", "standard")
        env.set_stage("diagnosis")
        return f"Triage complete. Urgency classified as: {urgency_level}", 0.0, False

    def _handle_ask_symptom(self, env: Any, payload: dict[str, Any]) -> Tuple[str, float, bool]:
        symptom = payload.get("symptom_id") or payload.get("symptom")
        if not symptom:
            return "Missing 'symptom' parameter in payload", -2.0, False
        
        env.record_symptom_queried(symptom)
        status = env.patient.respond_symptom(symptom)
        msg = f"Symptom {symptom}: {status}"
        return msg, 0.0, False

    def _handle_order_test(self, env: Any, payload: dict[str, Any]) -> Tuple[str, float, bool]:
        test = payload.get("test")
        if not test:
            return "Missing 'test' parameter in payload", -2.0, False
            
        if env.test_budget_remaining <= 0:
            return "Test budget exceeded", -2.0, False

        env.decrement_test_budget()
        env.record_test_ordered(test)

        # In DDxPlus, both symptoms and tests are encoded in evidences
        status = env.patient.respond_symptom(test)
        if status == "unknown":
            msg = f"Test result unavailable for {test}"
        else:
            msg = f"Test result for {test}: {status}"
            
        return msg, -1.0, False

    def _handle_give_diagnosis(self, env: Any, payload: dict[str, Any]) -> Tuple[str, float, bool]:
        diagnosis = payload.get("diagnosis")
        confidence = payload.get("confidence", 1.0)
        if not diagnosis:
            return "Missing 'diagnosis' parameter", -2.0, False
            
        env.record_diagnosis(diagnosis)
        env.set_diagnosis_confidence(confidence)
        env.set_stage("treatment")
        return f"Diagnosis submitted: {diagnosis} (confidence: {confidence})", 0.0, False

    def _handle_prescribe(self, env: Any, payload: dict[str, Any]) -> Tuple[str, float, bool]:
        drug = payload.get("drug_id") or payload.get("drug")
        dose = payload.get("dose", "Unknown dose")
        duration = payload.get("duration", "unspecified")
        if not drug:
            return "Missing 'drug' or 'drug_id' parameter", -2.0, False
            
        env.record_treatment(drug, dose, duration)
        env.set_stage("follow-up")
        return f"Treatment prescribed: {drug} {dose} for {duration}", 0.0, False

    def _handle_schedule_followup(self, env: Any, payload: dict[str, Any]) -> Tuple[str, float, bool]:
        days = payload.get("days")
        if days is None:
            return "Missing 'days' parameter", -2.0, False
            
        env.record_followup(days)
        return f"Follow-up scheduled in {days} days", 0.0, True

    def _handle_refer(self, env: Any, payload: dict[str, Any]) -> Tuple[str, float, bool]:
        from agents.specialist_agent import SpecialistAgent
        
        specialty = payload.get("specialty", "general")
        reason = payload.get("reason", "unspecified")
        
        specialist = SpecialistAgent(env.current_case, specialty=specialty)
        diagnosis = specialist.provide_diagnosis()
        treatment = specialist.recommend_treatment()
        
        env.activate_specialist(diagnosis, treatment)
        env.set_stage("treatment")
        
        return (
            f"Patient referred to {specialty} specialist (reason: {reason}). "
            f"Specialist diagnosis: {diagnosis}"
        ), 0.0, False

    def _handle_flag_emergency(self, env: Any, payload: dict[str, Any]) -> Tuple[str, float, bool]:
        reason = payload.get("reason", "No reason provided")
        env.record_emergency_flagged()
        return f"Emergency flagged! Reason: {reason}", 0.0, True
