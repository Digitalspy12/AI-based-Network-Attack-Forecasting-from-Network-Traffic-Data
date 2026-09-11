"""Local Ollama AI Agent — DigitalSpy analyst assistant.

Architecture (IMPLEMENTATION.md §22–24):
    forecast model → structured JSON → local agent → tools → analyst explanation

Runtime: Ollama (qwen2.5:7b) via local REST API.
Orchestration: plain Python (no LangChain/LlamaIndex in Week-1).

CRITICAL CONSTRAINTS (§22, §23, §30):
    ✓ Agent reads structured outputs from the forecast model
    ✓ Agent calls approved tools only
    ✓ Agent may explain, summarise, and support what-if analysis
    ✗ Agent MUST NEVER rewrite or invent model probabilities
    ✗ Agent MUST NEVER fabricate evidence
    ✗ Agent MUST NEVER turn uncertainty into certainty
    ✗ Agent MUST NEVER execute destructive actions automatically
    ✗ No cloud APIs — Ollama only
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

# Ollama defaults
_OLLAMA_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b"
_TIMEOUT_SECONDS = 120

# System prompt — defines the agent's role and constraints
_SYSTEM_PROMPT = """You are DigitalSpy Analyst — an AI assistant for a SOC (Security Operations Centre).

You have access to structured outputs from a network attack forecast model.
Your role is to explain the model's predictions, provide cybersecurity context,
and help analysts understand what the data suggests.

STRICT RULES:
1. The forecast model is the NUMERICAL TRUTH. You MUST report its probabilities exactly as given.
2. You MUST NOT invent, modify, or fabricate any probability, score, or numeric value.
3. Use language like "the model estimates", "the forecast suggests", "evidence indicates".
4. You MUST NOT claim certainty — these are probabilistic forecasts.
5. You MUST NOT recommend autonomous blocking or destructive actions.
6. You MUST NOT use cloud APIs or external services.
7. You MAY explain ATT&CK context, suggest investigative steps, and summarise findings.

Output format: concise, structured analyst briefings. Use bullet points where appropriate."""


class DigitalSpyAgent:
    """Local Ollama-based analyst agent.

    Tool set (IMPLEMENTATION.md §23):
        get_recent_flows()
        get_host_history()
        lookup_attack_technique()
        explain_forecast()
        run_what_if()
        generate_incident_summary()
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _OLLAMA_BASE_URL,
        forecast_engine=None,
        state_store: Optional[dict] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.forecast_engine = forecast_engine
        self.state_store = state_store or {}  # host → list of windows
        self._verify_ollama()

    def _verify_ollama(self) -> None:
        """Check Ollama is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.model in m for m in models):
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {models}. Agent may fail."
                )
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable at %s. Agent calls will fail.", self.base_url)

    def _call_ollama(self, prompt: str, system: str = _SYSTEM_PROMPT) -> str:
        """Send a prompt to Ollama and return the response text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.3,   # Low temp for consistent, factual outputs
                "num_predict": 512,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return f"[Agent error: {e}]"

    # ── Tool implementations ────────────────────────────────────────────────

    def get_recent_flows(self, host_ip: str, n: int = 5) -> dict:
        """Return the N most recent windows for a host."""
        windows = self.state_store.get(host_ip, [])
        return {
            "host_ip": host_ip,
            "recent_windows": windows[-n:] if windows else [],
            "total_windows": len(windows),
        }

    def get_host_history(self, host_ip: str) -> dict:
        """Return full Z_t sequence for a host."""
        windows = self.state_store.get(host_ip, [])
        z_seq = [w.get("z_t", "UNKNOWN") for w in windows]
        return {
            "host_ip": host_ip,
            "z_t_sequence": z_seq,
            "sequence_length": len(z_seq),
        }

    def lookup_attack_technique(self, z_t_label: str) -> dict:
        """Query ATT&CK mapper for a Z_t bucket."""
        from digitalspy.attack_context.attck_mapper import build_security_context
        return build_security_context(z_t_label, risk_prob=0.5)

    def explain_forecast(self, forecast: dict, shap_explanation: dict) -> str:
        """Generate a natural language explanation of the forecast.

        The agent explains the model output — it does NOT modify probabilities.

        Args:
            forecast: Output dict from ForecastEngine.predict()
            shap_explanation: Output from shap_explainer.explain_lstm()

        Returns:
            Analyst explanation text.
        """
        prompt = f"""Analyse this network attack forecast and explain it to a SOC analyst.

FORECAST DATA (do not modify these values):
{json.dumps(forecast, indent=2)}

FEATURE EVIDENCE (SHAP):
{json.dumps(shap_explanation, indent=2)}

Please:
1. Summarise the risk trajectory over the next 50 seconds.
2. Identify the predicted attack tactic and its significance.
3. Name the top 3 most influential features and what they may indicate.
4. Note the most important historical window (highest attention weight) and its significance.
5. Recommend 2-3 specific investigative actions for the analyst.

Remember: report model probabilities exactly. Do not invent or modify any numbers."""

        return self._call_ollama(prompt)

    def run_what_if(
        self,
        baseline_forecast: dict,
        scenario_forecast: dict,
        modified_feature: str,
        original_value: float,
        new_value: float,
    ) -> str:
        """Explain a what-if scenario comparison.

        The LLM explains the comparison — it does NOT invent numerical outcomes.
        Both forecasts come from the actual model.

        Args:
            baseline_forecast: Original model output.
            scenario_forecast: Modified model output (from actual model run).
            modified_feature: Name of the feature that was changed.
            original_value: Original feature value.
            new_value: New feature value.

        Returns:
            Analyst explanation of the scenario difference.
        """
        prompt = f"""Compare these two network threat forecasts (baseline vs what-if scenario).

FEATURE MODIFIED: {modified_feature}
  Original value: {original_value:.4f}
  New value:      {new_value:.4f}

BASELINE FORECAST:
{json.dumps(baseline_forecast, indent=2)}

SCENARIO FORECAST (with modified feature):
{json.dumps(scenario_forecast, indent=2)}

Please:
1. Describe how the risk trajectory changed.
2. Identify whether the predicted tactic changed and why this matters.
3. Explain what modifying '{modified_feature}' represents in network behaviour terms.
4. Provide a concise threat assessment comparing baseline vs scenario.

Report the model values exactly as given. Do not invent any probabilities."""

        return self._call_ollama(prompt)

    def generate_incident_summary(
        self,
        host_ip: str,
        forecast: dict,
        shap_explanation: dict,
        attck_context: dict,
    ) -> str:
        """Generate a complete analyst incident summary.

        Args:
            host_ip: Source host IP address.
            forecast: ForecastEngine output.
            shap_explanation: SHAP explanation.
            attck_context: ATT&CK mapper output.

        Returns:
            Structured incident summary text.
        """
        prompt = f"""Generate a concise SOC incident summary for the following network threat detection.

HOST: {host_ip}

FORECAST:
{json.dumps(forecast, indent=2)}

FEATURE EVIDENCE:
{json.dumps(shap_explanation, indent=2)}

ATT&CK CONTEXT:
{json.dumps(attck_context, indent=2)}

Format the summary as:
## Incident Summary
**Host:** {host_ip}
**Threat Level:** [based on risk probabilities]
**Predicted Tactic:** [from forecast]
**ATT&CK Mapping:** [tactic and techniques]

## Evidence
[Key SHAP features and attention insights]

## Forecast Timeline
[Risk progression t+1 to t+5]

## Recommended Actions
[3-5 specific investigative steps]

## Limitations
[Brief note on model uncertainty]

Report all probabilities exactly as given by the model."""

        return self._call_ollama(prompt)
