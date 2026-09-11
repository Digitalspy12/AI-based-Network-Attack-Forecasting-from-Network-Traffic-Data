"""ATT&CK semantic context layer.

Maps DigitalSpy Z_t buckets to MITRE ATT&CK tactics and likely techniques.

IMPORTANT: The six project buckets are project-defined mappings from dataset
attack labels. They are NOT official ATT&CK ground-truth stages.

UI language:
    "Predicted ATT&CK tactic"   ✓
    "Likely ATT&CK technique"   ✓
    "Official ATT&CK ground truth" ✗
"""
from __future__ import annotations

from digitalspy import config


def get_attck_tactic(z_t_label: str) -> str | None:
    """Map a Z_t bucket to its corresponding ATT&CK tactic name."""
    mapper = config.labels()["attck_tactic_map"]
    return mapper.get(z_t_label)


def get_attck_techniques(z_t_label: str) -> list[str]:
    """Return likely ATT&CK techniques for a Z_t bucket."""
    mapper = config.labels()["attck_technique_map"]
    return mapper.get(z_t_label, [])


def build_security_context(z_t_label: str, risk_prob: float) -> dict:
    """Build full security context card for a predicted tactic.

    Args:
        z_t_label: Predicted Z_t bucket name.
        risk_prob: Model risk probability (float [0,1]).

    Returns:
        dict with tactic, techniques, risk_level, description.
    """
    tactic = get_attck_tactic(z_t_label)
    techniques = get_attck_techniques(z_t_label)

    if risk_prob >= 0.75:
        risk_level = "HIGH"
    elif risk_prob >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    descriptions = {
        "BENIGN": "No malicious activity detected in forecast window.",
        "RECON": "Adversary is likely conducting active network reconnaissance.",
        "INITIAL_ACCESS": "Adversary may be attempting to gain initial foothold.",
        "LATERAL_MOVEMENT": "Adversary may be moving laterally across the network.",
        "C2": "Adversary may have established Command and Control communication.",
        "IMPACT": "Adversary is likely conducting a denial-of-service or destructive attack.",
    }

    return {
        "z_t_bucket": z_t_label,
        "attck_tactic": tactic,
        "attck_techniques": techniques,
        "risk_level": risk_level,
        "risk_probability": round(risk_prob, 4),
        "description": descriptions.get(z_t_label, ""),
        "disclaimer": (
            "This prediction is based on statistical temporal patterns in "
            "network flow features. It is not causal proof of attacker intent."
        ),
    }
