from typing import Dict


KEYWORD_DIMENSION_MAP = {
    "uptime": "T",
    "system": "T",
    "availability": "T",
    "downtime": "T",
    "integration": "T",

    "error": "P",
    "processing": "P",
    "throughput": "P",
    "delay": "P",
    "cycle": "P",
    "time": "P",

    "ownership": "R",
    "role": "R",
    "accountability": "R",
    "handover": "R",
    "escalation": "R",

    "training": "A",
    "adoption": "A",
    "usage": "A",
    "communication": "A",
    "engagement": "A",
}


def suggest_dimension_for_kpi(kpi_name: str) -> Dict[str, str]:
    """
    Suggest a likely system dimension for a KPI name using simple keyword rules.
    """
    normalized = kpi_name.strip().lower()

    for keyword, dimension in KEYWORD_DIMENSION_MAP.items():
        if keyword in normalized:
            return {
                "dimension": dimension,
                "reason": f"Matched keyword '{keyword}' in KPI name.",
            }

    return {
        "dimension": "P",
        "reason": "No clear keyword match found. Defaulted to Process.",
    }