from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

class MetricInput(BaseModel):
    name: str
    value: Any


class AssessmentInput(BaseModel):
    company_id: str
    industry: Optional[str] = None
    target_level: int = 2
    assessed_at: Optional[str] = None
    metrics: List[MetricInput]

DIMENSION_LABELS = {
    "T": "Technologie",
    "P": "Prozesse",
    "R": "Rollen & Verantwortung",
    "A": "Adoption & Veränderungsbereitschaft",
}

RISK_LABELS = {
    "low":    "Gering – Übergang ist realistisch",
    "medium": "Mittel – gezielte Maßnahmen notwendig",
    "high":   "Hoch – signifikante Lücken müssen zuerst geschlossen werden",
}

SCORE_BAR = ["░░░░░░░░░░", "██░░░░░░░░", "█████░░░░░", "██████████"]

def _format_score(score: float) -> str:
    bar_index = min(int(round(score)), 3)
    return f"{SCORE_BAR[bar_index]} {score: .1f} / 3"

def _format_bottleneck(dimension: str, detail: Dict[str, Any]) -> str:
    label = DIMENSION_LABELS.get(dimension, dimension)
    issue = detail.get("issue", "")
    changes = detail.get("required_changes", [])
    capacities = detail.get("required_capacities", [])

    lines = [
        f"  Dimension: {label}",
        f"  Problem:   {issue}",
    ]
    if changes:
        lines.append(f"  Lücken:    {', '.join(changes)}")
    if capacities:
        lines.append(f"  Bedarf:    {', '.join(capacities)}")
    return "\n".join(lines)


def _format_maturity_overview(
    scores: Dict[str, float],
    descriptions: Dict[str, str],
) -> str:
    lines = []
    for dim, score in scores.items():
        label = DIMENSION_LABELS.get(dim, dim)
        desc = descriptions.get(dim, "")
        lines.append(f"  {label}: {_format_score(score)}")
        lines.append(f"    → {desc}")
    return "\n".join(lines)

SYSTEM_PROMPT = """Du bist ein erfahrener Unternehmensberater, spezialisiert auf \
organisatorische Reifegradmodelle und operative Transformation.

Deine Aufgabe ist es, auf Basis eines strukturierten Maturity-Assessments eine \
klare, priorisierte Handlungsempfehlung zu geben.

Deine Antwort folgt immer dieser Struktur:
1. Gesamteinschätzung (2–3 Sätze, direkt und ehrlich)
2. Top-3-Prioritäten (konkret, umsetzbar, priorisiert nach Impact)
3. Dimension mit dem größten Hebel (1 Absatz mit Begründung)
4. Risiko-Einschätzung (was passiert, wenn nichts getan wird)
5. Nächster konkreter Schritt (1 Aktion, sofort umsetzbar)

Stil:
- Keine Floskeln, kein "Es ist wichtig zu beachten, dass..."
- Direkt und handlungsorientiert, wie ein guter Berater im Meeting
- Zahlen und Scores zitieren, um Aussagen zu belegen
- Wenn RAG-Kontext vorhanden ist, diesen aktiv nutzen und referenzieren
"""
def build_user_prompt(
    assessment: AssessmentInput,
    engine_output: Dict[str, Any],
    rag_context: Optional[str] = None,
) -> str:
    scores = engine_output.get("dimension_scores", {})
    descriptions = engine_output.get("maturity_descriptions", {})
    overall = engine_output.get("overall_readiness", 0.0)
    bottlenecks = engine_output.get("bottlenecks", [])
    bottleneck_details = engine_output.get("bottleneck_details", {})
    feasible = engine_output.get("transition_feasible", False)
    risk = engine_output.get("transition_risk", "unknown")
    required_changes = engine_output.get("required_changes", {})

    date_str = assessment.assessed_at or datetime.today().strftime("%d.%m.%Y")
    industry_str = assessment.industry or "nicht angegeben"
    feasible_str = "Ja" if feasible else "Nein – Lücken müssen zuerst adressiert werden"
    risk_str = RISK_LABELS.get(risk, risk)

    # Bottleneck-Blöcke formatieren
    bottleneck_blocks = ""
    if bottleneck_details:
        blocks = [
            _format_bottleneck(dim, detail)
            for dim, detail in bottleneck_details.items()
        ]
        bottleneck_blocks = "\n\n".join(blocks)
    else:
        bottleneck_blocks = "  Keine kritischen Bottlenecks – alle Dimensionen erreichen das Ziel-Level."

    # RAG-Kontext einfügen (optional)
    rag_block = ""
    if rag_context and rag_context.strip():
        rag_block = f"""
── REFERENZKONTEXT (aus internen Dokumenten / Benchmarks) ──────────────────────
{rag_context.strip()}
────────────────────────────────────────────────────────────────────────────────
"""
        prompt = f"""
        ── ASSESSMENT-KONTEXT ────────────────────────────────────────────────────────
        Unternehmen:   {assessment.company_id}
        Branche:       {industry_str}
        Datum:         {date_str}
        Ziel-Level:    {assessment.target_level} / 3
        Übergang mögl: {feasible_str}
        Gesamtrisiko:  {risk_str}

        ── REIFEGRAD-SCORES ─────────────────────────────────────────────────────────
        {_format_maturity_overview(scores, descriptions)}

          Gesamtbereitschaft (Minimum aller Dimensionen): {overall:.1f} / 3

        ── BOTTLENECKS ──────────────────────────────────────────────────────────────
        {bottleneck_blocks}
        {rag_block}
        ── AUFGABE ──────────────────────────────────────────────────────────────────
        Erstelle eine klare, priorisierte Handlungsempfehlung für das Unternehmen \
        "{assessment.company_id}" auf Basis der oben stehenden Daten.
        Folge dabei der vorgegebenen Struktur aus deinem System-Prompt.
        """.strip()

        return prompt

def build_prompt_messages(
        assessment: AssessmentInput,
        engine_output: Dict[str, Any],
        rag_context: Optional[str] = None,
        ) -> List[Dict[str, str]]:
        """
        Gibt eine Liste von Messages zurück, kompatibel mit OpenAI- und
        Anthropic-Chat-APIs:
          [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(assessment, engine_output, rag_context)},
        ]
