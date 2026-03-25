from typing import List, Dict, Any
from collections import defaultdict

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

def identify_bottlenecks(scores:Dict[str, float]) -> List[str]:
    if not scores:
        return []

    min_value = min(scores.values())
    return [dim for dim, val in scores.items() if val == min_value]

def check_transition_feasibility(scores: Dict[str, float], target_level: int) -> bool:
    return all(score >= target_level for score in scores.values())

def calculate_transition_risk(scores: Dict[str, float], target_level: int) -> str:
    gaps = [target_levelk - s for s in scores.values() if s < target_level]

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

    overall = calculate_overall_readiness(scores)

    bottlenecks = identify_bottlenecks(scores)

    feasible = check_transition_feasibility(scores, target_level)

    risk = calculate_transition_risk(scores, target_level)

    required_changes = build_required_changes(indicators, target_level)

    return {
        "dimension_scores": scores,
        "overall_readiness": overall,
        "bottlenecks": bottlenecks,
        "transition_feasible": feasible,
        "transition_risk": risk,
        "required_changes": required_changes,
    }