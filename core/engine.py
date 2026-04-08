"""
engine.py

Deterministic scoring engine for the Explainable AI Maturity Assessment System.

This module transforms mapped indicator values into a structured assessment result.

It is the core decision layer of the system and is responsible for:
- calculating dimension-level maturity scores
- identifying bottlenecks
- evaluating transition feasibility
- estimating transition risk
- deriving required changes and capacities
- generating explainable detail structures for downstream use

This deterministic layer is intentionally separated from the LLM layer.
The engine decides. The LLM explains.
"""

from collections import defaultdict
from typing import Any, Dict, List


# =============================================================================
# Static reference models
# =============================================================================
DIMENSION_LABELS = {
    "T": "Technology",
    "P": "Process",
    "R": "Responsibility",
    "A": "Adoption",
}

MATURITY_MODEL = {
    "T": {
        0: "No system support. Work is fully manual.",
        1: "Basic tools exist but are unstable or fragmented.",
        2: "Systems are reliable and support most operations.",
        3: "Systems are fully integrated, automated, and scalable.",
    },
    "P": {
        0: "No defined processes. Execution is ad hoc.",
        1: "Processes exist but are inconsistent and not followed.",
        2: "Processes are defined and mostly followed.",
        3: "Processes are standardised, optimised, and repeatable.",
    },
    "R": {
        0: "No clear ownership. Responsibilities are unclear.",
        1: "Ownership exists but is not consistently enforced.",
        2: "Responsibilities are mostly clear and assigned.",
        3: "Clear ownership with strong accountability and decision authority.",
    },
    "A": {
        0: "Strong resistance to change.",
        1: "Change is accepted reluctantly.",
        2: "Change is generally accepted.",
        3: "Change is actively supported and driven.",
    },
}

BOTTLENECK_ISSUES = {
    "T": "Technology support is insufficient for stable and scalable execution.",
    "R": "Responsibilities are not defined clearly enough for reliable ownership.",
    "P": "Processes are not consistent enough for efficient execution.",
    "A": "Organisational acceptance is too low to support adoption of change.",
}

CAPACITY_REQUIREMENTS = {
    "T": {
        "T1": "Higher automation capability",
        "T2": "More reliable system availability",
        "T3": "Stronger system integration",
    },
    "R": {
        "R1": "Clearer role definition",
        "R2": "Stronger ownership assignment",
        "R3": "Clearer decision authority",
    },
    "P": {
        "P1": "Faster process execution",
        "P2": "Lower error levels",
        "P3": "Higher process standardisation",
    },
    "A": {
        "A1": "Stronger change communication",
        "A2": "Higher training coverage",
        "A3": "Broader tool adoption",
    },
}


# =============================================================================
# Core scoring functions
# =============================================================================

def calculate_dimension_scores(indicators: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate indicator values into average dimension scores.

    Args:
        indicators: List of mapped indicator dictionaries.

    Returns:
        Dictionary of average scores by dimension.
    """
    grouped: Dict[str, List[float]] = defaultdict(list)

    for item in indicators:
        grouped[item["dimension"]].append(item["value"])

    scores: Dict[str, float] = {}
    for dim, values in grouped.items():
        scores[dim] = sum(values) / len(values)

    return scores


def calculate_overall_readiness(scores: Dict[str, float]) -> float:
    """
    Calculate the overall readiness score.

    The overall readiness is defined as the minimum score across all dimensions.
    This reflects the idea that the weakest dimension constrains the transition.

    Args:
        scores: Dictionary of dimension scores.

    Returns:
        Overall readiness score.
    """
    return min(scores.values()) if scores else 0.0


def identify_bottlenecks(scores: Dict[str, float], target_level: int) -> List[str]:
    """
    Identify all dimensions below the target maturity level.

    Args:
        scores: Dictionary of dimension scores.
        target_level: Required target maturity level.

    Returns:
        List of bottleneck dimension keys.
    """
    return [dim for dim, val in scores.items() if val < target_level]


def check_transition_feasibility(scores: Dict[str, float], target_level: int) -> bool:
    """
    Check whether the transition is feasible.

    A transition is considered feasible only if all dimensions reach or exceed
    the required target level.

    Args:
        scores: Dictionary of dimension scores.
        target_level: Required target maturity level.

    Returns:
        True if feasible, otherwise False.
    """
    return all(score >= target_level for score in scores.values())


def calculate_transition_risk(scores: Dict[str, float], target_level: int) -> str:
    """
    Estimate transition risk based on maturity gaps.

    The larger the maximum gap between current score and target level,
    the higher the transition risk.

    Args:
        scores: Dictionary of dimension scores.
        target_level: Required target maturity level.

    Returns:
        Risk label: 'low', 'medium', or 'high'.
    """
    gaps = [target_level - s for s in scores.values() if s < target_level]

    if not gaps:
        return "low"

    max_gap = max(gaps)

    if max_gap >= 1.5:
        return "high"
    if max_gap >= 0.5:
        return "medium"
    return "low"


# =============================================================================
# Explainability helpers
# =============================================================================

def build_required_changes(indicators: List[Dict[str, Any]], target_level: int) -> Dict[str, List[str]]:
    """
    Identify which indicators remain below the target level.

    Args:
        indicators: List of mapped indicators.
        target_level: Required target maturity level.

    Returns:
        Dictionary mapping each dimension to its missing indicator names.
    """
    changes: Dict[str, List[str]] = defaultdict(list)

    for item in indicators:
        if item["value"] < target_level:
            changes[item["dimension"]].append(item["indicator"])

    return dict(changes)


def build_required_capacities(required_changes: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Translate missing indicators into missing organisational capacities.

    Args:
        required_changes: Dictionary of indicators below target level.

    Returns:
        Dictionary of required capacities by dimension.
    """
    capacities: Dict[str, List[str]] = {}

    for dimension, indicators in required_changes.items():
        mapped = [
            CAPACITY_REQUIREMENTS.get(dimension, {}).get(indicator, indicator)
            for indicator in indicators
        ]
        capacities[dimension] = list(dict.fromkeys(mapped))

    return capacities


def build_bottleneck_details(
    bottlenecks: List[str],
    required_changes: Dict[str, List[str]],
    required_capacities: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a structured detail block for each bottleneck.

    Args:
        bottlenecks: List of bottleneck dimensions.
        required_changes: Missing indicators by dimension.
        required_capacities: Missing capacities by dimension.

    Returns:
        Dictionary of detailed bottleneck information.
    """
    details: Dict[str, Dict[str, Any]] = {}

    for dimension in bottlenecks:
        details[dimension] = {
            "issue": BOTTLENECK_ISSUES.get(dimension, "Structural limitation detected."),
            "required_changes": required_changes.get(dimension, []),
            "required_capacities": required_capacities.get(dimension, []),
        }

    return details


def add_maturity_descriptions(scores: Dict[str, float]) -> Dict[str, str]:
    """
    Add human-readable maturity descriptions for each dimension score.

    Scores are rounded to the nearest maturity level for text description.

    Args:
        scores: Dictionary of dimension scores.

    Returns:
        Dictionary of maturity description strings by dimension.
    """
    descriptions: Dict[str, str] = {}

    for dim, score in scores.items():
        rounded = int(round(score))
        descriptions[dim] = MATURITY_MODEL.get(dim, {}).get(rounded, "")

    return descriptions

def classify_dimensions(dimension_scores: dict[str, float]):
    strengths = []
    weaknesses = []
    neutral = []

    for dimension, score in dimension_scores.items():
        if score <= 1.3:
            weaknesses.append(dimension)
        elif score >= 2.0:
            strengths.append(dimension)
        else:
            neutral.append(dimension)

    return strengths, weaknesses, neutral

def derive_cross_dimension_insights(strengths: List[str], weaknesses: List[str]) -> List[str]:
    """
    Derive strategic cross-dimension insights.

    This links strong dimensions with weak ones to identify leverage effects.
    """
    insights = []

    if "A" in strengths and "T" in weaknesses:
        insights.append(
            "Acceptance is already strong and can be used to accelerate technology adoption and implementation."
        )

    if "P" in strengths and "T" in weaknesses:
        insights.append(
            "Stable processes can reduce implementation risk while technology capabilities are upgraded."
        )

    if "R" in strengths and "A" in weaknesses:
        insights.append(
            "Clear ownership can be used to strengthen acceptance and alignment during change."
        )

    if "R" in strengths and "P" in weaknesses:
        insights.append(
            "Strong ownership can help enforce more consistent process execution."
        )

    if "T" in strengths and "A" in weaknesses:
        insights.append(
            "Existing technology strength can support adoption if communication and enablement are improved."
        )

    return insights


def build_executive_summary(result: Dict[str, Any]) -> str:
    weakest_key = result["weaknesses"][0] if result.get("weaknesses") else None
    strongest_key = result["strengths"][0] if result.get("strengths") else None

    weakest = DIMENSION_LABELS.get(weakest_key, "no major constraint")
    strongest = DIMENSION_LABELS.get(strongest_key, "no strong leverage")
    readiness = result.get("overall_readiness", 0)

    return f"""
The current system shows an overall readiness of {readiness:.1f}.

The primary constraint lies in {weakest}, limiting reliable execution and scalability.
At the same time, {strongest} represents a key strength that can be leveraged to accelerate improvements.

Focusing on resolving the main constraint while utilising existing strengths will enable faster and more sustainable system development.
""".strip()


def build_leverage_explanation(result: Dict[str, Any]) -> str:
    if result.get("cross_dimension_insights"):
        return result["cross_dimension_insights"][0]

    if result.get("strengths"):
        strongest_key = result["strengths"][0]
        strongest = DIMENSION_LABELS.get(strongest_key, strongest_key)
        return f"{strongest} can be used as the main leverage point to stabilise weaker dimensions and support the transition."

    return "No clear leverage point has been identified yet."

# =============================================================================
# Main orchestration function
# =============================================================================

def run_deterministic_engine(indicators: List[Dict[str, Any]], target_level: int) -> Dict[str, Any]:
    """
    Run the full deterministic assessment pipeline.

    This function combines all scoring and explainability steps into one
    structured result object used by the API and the LLM explanation layer.

    Args:
        indicators: List of mapped indicators.
        target_level: Required target maturity level.

    Returns:
        Structured engine result dictionary.
    """
    scores = calculate_dimension_scores(indicators)
    overall = calculate_overall_readiness(scores)
    bottlenecks = identify_bottlenecks(scores, target_level)
    feasible = check_transition_feasibility(scores, target_level)
    risk = calculate_transition_risk(scores, target_level)
    required_changes = build_required_changes(indicators, target_level)
    required_capacities = build_required_capacities(required_changes)
    bottleneck_details = build_bottleneck_details(
        bottlenecks,
        required_changes,
        required_capacities,
    )
    maturity_descriptions = add_maturity_descriptions(scores)
    strengths, weaknesses, neutral = classify_dimensions(scores)
    cross_dimension_insights = derive_cross_dimension_insights(strengths, weaknesses)

    result = {
        "dimension_scores": scores,
        "maturity_descriptions": maturity_descriptions,
        "overall_readiness": overall,
        "bottlenecks": bottlenecks,
        "transition_feasible": feasible,
        "transition_risk": risk,
        "required_changes": required_changes,
        "required_capacities": required_capacities,
        "bottleneck_details": bottleneck_details,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "neutral": neutral,
        "cross_dimension_insights": cross_dimension_insights,
    }

    result["executive_summary"] = build_executive_summary(result)
    result["leverage_explanation"] = build_leverage_explanation(result)

    return result