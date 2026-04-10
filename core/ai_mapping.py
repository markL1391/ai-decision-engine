from typing import Dict, Optional
import os


KEYWORD_DIMENSION_MAP = {
    "uptime": "T",
    "system": "T",
    "availability": "T",
    "downtime": "T",
    "integration": "T",
    "automated": "T",
    "infrastructure": "T",
    "software": "T",
    "hardware": "T",
    "api": "T",
    "server": "T",

    "error": "P",
    "processing": "P",
    "throughput": "P",
    "delay": "P",
    "cycle": "P",
    "time": "P",
    "speed": "P",
    "efficiency": "P",
    "quality": "P",
    "defect": "P",
    "resolution": "P",

    "ownership": "R",
    "role": "R",
    "accountability": "R",
    "handover": "R",
    "escalation": "R",
    "responsibility": "R",
    "team": "R",
    "sla": "R",
    "agreement": "R",

    "training": "A",
    "adoption": "A",
    "usage": "A",
    "communication": "A",
    "engagement": "A",
    "change": "A",
    "resistance": "A",
    "satisfaction": "A",
    "feedback": "A",
}

DIMENSION_LABELS = {
    "T": "Technologie",
    "P": "Prozess",
    "R": "Verantwortung",
    "A": "Akzeptanz",
}

DIMENSION_LABELS_EN = {
    "T": "Technology",
    "P": "Process",
    "R": "Responsibility",
    "A": "Acceptance",
}


def suggest_dimension_for_kpi(kpi_name: str, language: str = "de") -> Dict[str, str]:
    """
    Suggest a likely system dimension for a KPI name using simple keyword rules.
    Falls OPENAI_API_KEY gesetzt ist, wird OpenAI für bessere Vorschläge genutzt.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return _suggest_with_openai(kpi_name, api_key, language)
    except Exception:
        pass
    
    return _suggest_with_keywords(kpi_name, language)


def _suggest_with_keywords(kpi_name: str, language: str = "de") -> Dict[str, str]:
    """
    Suggest dimension using keyword matching (fallback).
    """
    normalized = kpi_name.strip().lower()
    labels = DIMENSION_LABELS if language == "de" else DIMENSION_LABELS_EN

    for keyword, dimension in KEYWORD_DIMENSION_MAP.items():
        if keyword in normalized:
            return {
                "dimension": dimension,
                "dimension_label": labels.get(dimension, dimension),
                "reason": f"Matched keyword '{keyword}' in KPI name." if language == "en" else f"Stichwort '{keyword}' im KPI-Namen erkannt.",
            }

    return {
        "dimension": "P",
        "dimension_label": labels["P"],
        "reason": "No clear keyword match found. Defaulted to Process." if language == "en" else "Kein eindeutiges Stichwort gefunden. Standard: Prozess.",
    }


def _suggest_with_openai(kpi_name: str, api_key: str, language: str = "de") -> Dict[str, str]:
    """
    Use OpenAI to suggest the best dimension for a KPI with contextual reasoning.
    """
    from openai import OpenAI
    
    labels = DIMENSION_LABELS if language == "de" else DIMENSION_LABELS_EN
    
    system_prompt = """Du bist ein Mapping-Experte für operative KPIs.
Gib für den gegebenen KPI-Namen die wahrscheinlichste Dimension zurück:
- T (Technology): Systeme, Infrastruktur, Automatisierung, Verfügbarkeit
- P (Process): Geschwindigkeit, Qualität, Fehler, Durchsatz, Effizienz
- R (Responsibility): Rollen, Ownership, Übergaben, SLA, Verantwortlichkeiten
- A (Acceptance): Training, Nutzung, Kommunikation, Veränderungsbereitschaft

Antworte im Format: DIMENSION|XEGRÜNDUNG
z.B.: P|Die Bearbeitungszeit ist ein klassischer Prozess-KPI"""
    
    user_prompt = f"KPI: {kpi_name}"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=150,
        )
        
        result = response.choices[0].message.content.strip()
        
        if "|" in result:
            dimension, reason = result.split("|", 1)
            dimension = dimension.strip()
            reason = reason.strip()
            
            if dimension in labels:
                return {
                    "dimension": dimension,
                    "dimension_label": labels[dimension],
                    "reason": reason,
                    "source": "ai",
                }
        
        return _suggest_with_keywords(kpi_name, language)
        
    except Exception:
        return _suggest_with_keywords(kpi_name, language)