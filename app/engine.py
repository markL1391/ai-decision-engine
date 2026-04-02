from typing import List, Dict, Any
from collections import defaultdict

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


def calculate_dimension_scores(indicators: List[Dict[str, Any]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)

    for item in indicators:
        grouped[item["dimension"]].append(item["value"])

    scores: Dict[str, float] = {}
    for dim, values in grouped.items():
        scores[dim] = sum(values) / len(values)

    return scores


def calculate_overall_readiness(scores: Dict[str, float]) -> float:
    return min(scores.values()) if scores else 0.0


def identify_bottlenecks(scores: Dict[str, float], target_level: int) -> List[str]:
    return [dim for dim, val in scores.items() if val < target_level]


def check_transition_feasibility(scores: Dict[str, float], target_level: int) -> bool:
    return all(score >= target_level for score in scores.values())


def calculate_transition_risk(scores: Dict[str, float], target_level: int) -> str:
    gaps = [target_level - s for s in scores.values() if s < target_level]

    if not gaps:
        return "low"

    max_gap = max(gaps)

    if max_gap >= 1.5:
        return "high"
    if max_gap >= 0.5:
        return "medium"
    return "low"


def build_required_changes(indicators: List[Dict[str, Any]], target_level: int) -> Dict[str, List[str]]:
    changes: Dict[str, List[str]] = defaultdict(list)

    for item in indicators:
        if item["value"] < target_level:
            changes[item["dimension"]].append(item["indicator"])

    return dict(changes)


def build_required_capacities(required_changes: Dict[str, List[str]]) -> Dict[str, List[str]]:
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
    details: Dict[str, Dict[str, Any]] = {}

    for dimension in bottlenecks:
        details[dimension] = {
            "issue": BOTTLENECK_ISSUES.get(dimension, "Structural limitation detected."),
            "required_changes": required_changes.get(dimension, []),
            "required_capacities": required_capacities.get(dimension, []),
        }

    return details


def add_maturity_descriptions(scores: Dict[str, float]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}

    for dim, score in scores.items():
        rounded = int(round(score))
        descriptions[dim] = MATURITY_MODEL.get(dim, {}).get(rounded, "")

    return descriptions


def run_deterministic_engine(indicators: List[Dict[str, Any]], target_level: int) -> Dict[str, Any]:
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

    return {
        "dimension_scores": scores,
        "maturity_descriptions": maturity_descriptions,
        "overall_readiness": overall,
        "bottlenecks": bottlenecks,
        "transition_feasible": feasible,
        "transition_risk": risk,
        "required_changes": required_changes,
        "required_capacities": required_capacities,
        "bottleneck_details": bottleneck_details,
    }