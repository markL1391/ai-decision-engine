from typing import List, Dict, Any
from collections import defaultdict

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
    grouped = defaultdict(list)

    for item in indicators:
        grouped[item["dimension"]].append(item["value"])

    scores = {}
    for dim, values in grouped.items():
        scores[dim] = sum(values) / len(values)

    return scores

def calculate_overall_readiness(scores: Dict[str, float]) -> float:
    return min(scores.values()) if scores else 0.0

def identify_bottlenecks(scores: Dict[str, float], target_level: int) -> List[str]:
    return [dim for dim, val in scores.items() if val < target_level]

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

def check_transition_feasibility(scores: Dict[str, float], target_level: int) -> bool:
    return all(score >= target_level for score in scores.values())

def calculate_transition_risk(scores: Dict[str, float], target_level: int) -> str:
    gaps = [target_level - s for s in scores.values() if s < target_level]

    if not gaps:
        return "low"

    max_gap = max(gaps)

    if max_gap >= 1.5:
        return "high"
    elif max_gap >= 0.5:
        return "medium"
    else:
        return "low"

def build_required_changes(indicators: List[Dict[str, Any]], target_level: int) -> Dict[str, List[str]]:
    changes = defaultdict(list)

    for item in indicators:
        if item["value"] < target_level:
            changes[item["dimension"]].append(item["indicator"])

    return dict(changes)

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

    return {
        "dimension_scores": scores,
        "overall_readiness": overall,
        "bottlenecks": bottlenecks,
        "transition_feasible": feasible,
        "transition_risk": risk,
        "required_changes": required_changes,
        "required_capacities": required_capacities,
        "bottleneck_details": bottleneck_details,
    }

def build_required_capacities(required_changes: Dict[str, List[str]]) -> Dict[str, List[str]]:
    capacities: Dict[str, List[str]] = {}

    for dimension, indicators in required_changes.items():
        mapped = [
            CAPACITY_REQUIREMENTS.get(dimension, {}).get(indicator, indicator)
            for indicator in indicators
        ]
        capacities[dimension] = mapped

    return capacities