from typing import Any, Dict, List


def score_custom_kpi(value: float, thresholds: List[float], direction: str) -> int:
    """
    Score a custom KPI on a 0-3 scale.

    For higher_better:
        >= thresholds[0] -> 3
        >= thresholds[1] -> 2
        >= thresholds[2] -> 1
        else -> 0

    For lower_better:
        <= thresholds[0] -> 3
        <= thresholds[1] -> 2
        <= thresholds[2] -> 1
        else -> 0
    """
    if len(thresholds) != 3:
        raise ValueError("thresholds must contain exactly 3 values")

    if direction == "higher_better":
        if value >= thresholds[0]:
            return 3
        if value >= thresholds[1]:
            return 2
        if value >= thresholds[2]:
            return 1
        return 0

    if direction == "lower_better":
        if value <= thresholds[0]:
            return 3
        if value <= thresholds[1]:
            return 2
        if value <= thresholds[2]:
            return 1
        return 0

    raise ValueError("direction must be 'higher_better' or 'lower_better'")


def map_custom_kpis_to_indicators(custom_kpis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert user-defined KPIs into engine-ready indicator records.
    """
    indicators: List[Dict[str, Any]] = []

    for kpi in custom_kpis:
        score = score_custom_kpi(
            value=float(kpi["value"]),
            thresholds=kpi["thresholds"],
            direction=kpi["direction"],
        )

        indicators.append({
            "dimension": kpi["dimension"],
            "indicator": kpi["name"],
            "value": score,
            "source_metric": kpi["name"],
        })

    return indicators