from typing import List, Dict, Any

def score_by_thresholds_desc(value: float, thresholds: List[tuple[float, int]]) -> int:
    """
    For metrics where a higher value is better.
    Example:
        [(80, 30), (50, 2), (20, 1)] means:
        >=80 -> 3, >=50 -> 2, >=20 -> 1, else 0
    """
    for threshold, score in thresholds:
        if value >= threshold:
            return score
    return 0

def score_by_thresholds_asc(value: float, thresholds: List[tuple[float, int]]) -> int:
    """
    For metrics where a lower value is better.
    Example:
    [(2, 3), (5, 2), (10, 1)] means:
    <=2 -> 3, <=5 -> 2, <=10 -> 1, else 0
    """
    for threshold, score in thresholds:
        if value <= threshold:
            return score
    return 0

def score_process_standardization(value: str) -> int:
    normalized = value.strip().lower()

    mapping = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "fully_standardized": 3,
        "fully standardized": 3,
    }
    return mapping.get(normalized, 0)

def score_role_clarity(value: str) -> int:
    normalized = value.strip().lower()

    mapping = {
        "unclear": 0,
        "partial": 1,
        "clear": 2,
        "fully_defined": 3,
        "fully defined": 3,
    }
    return mapping.get(normalized, 0)

def score_ownership_definition(value: str) -> int:
    normalized = value.strip().lower()

    mapping = {
        "none": 0,
        "informal": 1,
        "formal": 2,
        "enforced": 3,
    }
    return mapping.get(normalized, 0)

def map_single_metric(name: str, value: Any) -> List[Dict[str, Any]]:
    """
    Maps a single KPI to one or more indicators.
    Returns a list because one metric could theoretically affect multiple indicators.
    """
    indicators: List[Dict[str, Any]] = []

    if name == "automation_rate":
        indicators.append({
            "dimension": "T",
            "indicator": "T1",
            "value": score_by_thresholds_desc(float(value), [(80, 3), (50, 2), (20,1)]),
            "source_metric": name,
        })

    elif name == "system_availability":
        indicators.append({
            "dimension": "T",
            "indicator": "T2",
            "value": score_by_thresholds_desc(float(value), [(99, 3), (95, 2), (90, 1)]),
            "source_metric": name,
        })

    elif name == "error_rate":
        indicators.append({
            "dimension": "P",
            "indicator": "P2",
            "value": score_by_thresholds_asc(float(value), [(2, 3), (5, 2), (10, 1)]),
            "source_metric": name,
        })

    elif name == "order_processing_time":
        indicators.append({
            "dimension": "P",
            "indicator": "P1",
            "value": score_by_thresholds_asc(float(value), [(10, 3), (20, 2), (30, 1)]),
            "source_metric": name,
        })

    elif name == "process_standardization":
        indicators.append({
            "dimension": "P",
            "indicator": "P3",
            "value": score_process_standardization(str(value)),
            "source_metric": name,
        })

    elif name == "role_clarity":
        indicators.append({
            "dimension": "R",
            "indicator": "R1",
            "value": score_role_clarity(str(value)),
            "source_metric": name,
        })

    elif name == "ownership_definition":
        indicators.append({
            "dimension": "R",
            "indicator": "R2",
            "value": score_ownership_definition(str(value)),
            "source_metric": name,
        })

    elif name == "training_coverage":
        indicators.append({
            "dimension": "A",
            "indicator": "A2",
            "value": score_by_thresholds_desc(float(value), [(80, 3), (60, 2), (30, 1)]),
            "source_metric": name,
        })

    elif name == "tool_adoption":
        indicators.append({
            "dimension": "A",
            "indicator": "A3",
            "value": score_by_thresholds_desc(float(value), [(80, 3), (60, 2), (30, 1)]),
            "source_metric": name,
        })

    elif name == "change_communication":
        mapping = {
            "none": 0,
            "irregular": 1,
            "structured": 2,
            "embedded": 3,
        }
        indicators.append({
            "dimension": "A",
            "indicator": "A1",
            "value": mapping.get(str(value).strip().lower(), 0),
            "source_metric": name,
        })

    return indicators

def map_metrics_to_indicators(metrics: List[Any]) -> List[Dict[str, Any]]:
    """
    Accepts a list of MetricInput-like objects with .name and .value
    and returns mapped indicator dictionaries.
    """
    mapped_indicators: List[Dict[str, Any]] = []

    for metric in metrics:
        mapped_indicators.extend(map_single_metric(metric.name, metric.value))

    return mapped_indicators