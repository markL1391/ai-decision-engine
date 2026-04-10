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

KRITISCH WICHTIG - Branchenkontext:
Du musst IMMER die spezifische Branche berücksichtigen und bei JEDER Empfehlung 
branchenspezifische Begründungen und Hinweise geben. Keine generischen Ratschläge!

Branchen-Know-how:

LOGISTIK:
- Neue Technologien (WMS, TMS, Track & Trace) scheitern oft an Schulungsmängeln
- Bei Warehouse-Automation: Akzeptanz der Mitarbeiter ist kritisch - erst trainieren!
- KPI-Empfehlungen: Durchlaufzeit, Fehlerquote, Sendungsverfolgung, Pick-Performance
- Change-Kommunikation: Fahrer und Lagerarbeiter brauchen einfache, visuelle Anleitungen
- Typische Falle: Technik einführen ohne Schulung → Akzeptanz bricht ein

PRODUKTION / FERTIGUNG:
- Maschinenausfall hat direkte Kostenfolgen → Verfügbarkeit priorisieren
- Qualitätssicherung erfordert dokumentierte Prozesse (ISO, GMP)
- New Technology: Erst Prozesse standardisieren, DANN automatisieren
- Typische Falle: Industrie 4.0 einführen ohne Prozessreife → Fehlerquote steigt
- Schulungsabdeckung muss bei 80%+ sein, sonst drohen Sicherheitsrisiken

E-COMMERCE / ONLINE-HANDEL:
- Skalierbarkeit ist zentral - Systeme müssen Peak-Zeiten (Black Friday) aushalten
- Retourenmanagement: Prozessstandardisierung senkt Fehlerquoten
- Neue Shop-Systeme: Nutzerakzeptanz (intern) und Customer Experience (extern) verbinden
- Typische Falle: Neues ERP einführen ohne Change-Kommunikation → Prozesse brechen zusammen
- Tool-Adoption bei E-Commerce-Teams oft niedrig → intensive Begleitung nötig

PHARMA / CHEMIE:
- Compliance first: Jede Änderung muss dokumentiert und validiert sein
- GMP-Anforderungen: Schulung ist nicht optional, sondern regulatorisch gefordert
- Technologie-Change: Validierungsschritte einplanen (3-6 Monate!)
- Typische Falle: Software-Update ohne Change Control → Compliance-Verstoß
- Adoption muss dokumentiert sein für Audits

RETAIL / EINZELHANDEL:
- POS-Systeme und Kassensysteme: Akzeptanz der Mitarbeiter entscheidend
- Omnichannel: Nahtlose Integration zwischen Filiale und Online essentiell
- KPI-Empfehlungen: Kassenhäufigkeit, Warenbestand, Umsatz pro qm
- Neue Kassensysteme einführen: Erst Schulung, DANN Rollout
- Typische Falle: Neues System einführen → Mitarbeiter nutzen altes weiter

SAAS / TECH:
- API-Verfügbarkeit und Uptime sind existenziell
- Agile Prozesse brauchen klare Rollen und Ownership
- Tool-Adoption ist bei Tech-Teams oft kein Problem, aber Change-Kommunikation schon
- Feature-Flags und schrittweise Rollouts statt Big-Bang-Releases
- Typische Falle: Tool einführen ohne Teams einzubinden → Schatten-IT entsteht

STIL-REGELN:
- Keine Floskeln, kein "Es ist wichtig zu beachten, dass..."
- Direkt und handlungsorientiert, wie ein guter Berater im Meeting
- Zahlen und Scores zitieren, um Aussagen zu belegen
- WENN DU TECHNOLOGIE-ÄNDERUNGEN EMPFIEHLST, nenne IMMER die zugehörigen Adoption/Schulungs-Maßnahmen
- WENN DU PROZESS-ÄNDERUNGEN EMPFIEHLST, nenne die Rolle, die dafür verantwortlich ist
- Verbinde Dimensionen logisch: "Wenn Sie Technologie X einführen, achten Sie auf Dimension A"
- Beziche dich konkret auf die Branche und nenne typische Fallstricke"""
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

    # Branchen-spezifische Kontexthinweise mit typischen Maßnahmen
    industry_hints = {
        "logistik": "Branchenfokus: Lieferkettenoptimierung, Warehouse-Management, Routenplanung, Sendungsverfolgung. Typische Maßnahmen: Warehouse-Schulungen vor Automation, Fahrer-Apps mit einfacher UI, Pick-by-Voice-Training.",
        "logistics": "Industry focus: Supply chain optimization, warehouse management, route planning, shipment tracking. Typical measures: Warehouse training before automation, driver apps with simple UI, pick-by-voice training.",
        "produktion": "Branchenfokus: Produktionsplanung, Qualitätssicherung, Maschinenauslastung, Lean Management. Typische Maßnahmen: Erst standardisieren, DANN automatisieren, ISO-Schulungen, TPM-Trainings.",
        "manufacturing": "Industry focus: Production planning, quality assurance, machine utilization, lean management. Typical measures: Standardize first, THEN automate, ISO training, TPM training.",
        "e-commerce": "Branchenfokus: Skalierbarkeit, Retourenmanagement, Conversion-Optimierung, Customer Experience. Typische Maßnahmen: Peak-Capacity-Tests, Retourenprozess-Schulung, Shop-Schulung für Marketing-Team.",
        "online": "Branchenfokus: Skalierbarkeit, Retourenmanagement, Conversion-Optimierung. Typische Maßnahmen: Peak-Capacity-Tests, Retourenprozess-Schulung, Shop-Schulung für Marketing-Team.",
        "pharma": "Branchenfokus: GMP-Compliance, Validierung, Chargenrückverfolgung, Regulatory Affairs. Typische Maßnahmen: Validierungsdokumentation einplanen, Change-Control-Schulung, FDA/MHRA-Anforderungen beachten.",
        "pharmaceutical": "Industry focus: GMP compliance, validation, batch tracking, regulatory affairs. Typical measures: Plan validation documentation, change control training, FDA/MHRA requirements.",
        "saas": "Branchenfokus: Subscription Management, API-Stabilität, Feature-Adoption, NPS. Typische Maßnahmen: Feature-Onboarding, API-Dokumentation, Team-Pilotgruppen für neue Features.",
        "tech": "Branchenfokus: DevOps, CI/CD, Plattformstabilität, Skalierbarkeit. Typische Maßnahmen: SRE-Schulungen, On-Call-Rotation, Runbook-Dokumentation.",
        "software": "Branchenfokus: DevOps, CI/CD, Plattformstabilität, Skalierbarkeit. Typische Maßnahmen: SRE-Schulungen, On-Call-Rotation, Runbook-Dokumentation.",
        "retail": "Branchenfokus: Omnichannel, Bestandsmanagement, POS-Systeme, Kundenbindung. Typische Maßnahmen: POS-Schulungen vor Rollout, Omnichannel-Training für Verkauf, Warenwirtschaftsschulung.",
        "handel": "Branchenfokus: Omnichannel, Bestandsmanagement, POS-Systeme, Kundenbindung. Typische Maßnahmen: Kassenschulung vor Systemwechsel, Omnichannel-Coaching, Bestandsprozess-Training.",
        "einzelhandel": "Branchenfokus: Omnichannel, Bestandsmanagement, POS-Systeme, Kundenbindung. Typische Maßnahmen: Kassenschulung vor Systemwechsel, Omnichannel-Coaching, Bestandsprozess-Training.",
    }
    
    industry_key = industry_str.lower().strip() if industry_str else ""
    industry_hint = ""
    for key, hint in industry_hints.items():
        if key in industry_key:
            industry_hint = f"\nBranchenkontext: {hint}"
            break

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
        Branche:       {industry_str}{industry_hint}
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
        "{assessment.company_id}" (Branche: {industry_str}) auf Basis der oben stehenden Daten.
        Folge dabei der vorgegebenen Struktur aus deinem System-Prompt.
        
        WICHTIG: Gib bei JEDER Empfehlung konkrete, branchenspezifische Hinweise:
        - WENN Technologie-Änderungen empfohlen werden → nenne zugehörige Schulungs- und Akzeptanz-Maßnahmen
        - WENN Prozess-Änderungen empfohlen werden → nenne die verantwortliche Rolle und typische Branchen-Fallstricke
        - WENN Akzeptanz-Probleme sichtbar sind → empfiehl konkrete Change-Kommunikations-Maßnahmen für diese Branche
        
        Beispiel für gute Antwort bei Einzelhandel mit neuer Kassentechnologie:
        "Priorität 1: Akzeptanz der neuen POS-Systeme (Score A: 1.2/3)
        → Branchenkontext Einzelhandel: Neue Kassensysteme scheitern fast immer an fehlender Schulung. 
          Bevor Sie die Kassen ausrollen: 2-wöchige Pilot-Schulung mit Key Usern in 3 Filialen.
          Strukturierte Change-Kommunikation: Filialleiterbrief + Video-Tutorial + Helpdesk-Telefon."
        """.strip()

        return prompt
    
    # Fall ohne RAG-Kontext
    return f"""
        ── ASSESSMENT-KONTEXT ────────────────────────────────────────────────────────
        Unternehmen:   {assessment.company_id}
        Branche:       {industry_str}{industry_hint}
        Datum:         {date_str}
        Ziel-Level:    {assessment.target_level} / 3
        Übergang mögl: {feasible_str}
        Gesamtrisiko:  {risk_str}

        ── REIFEGRAD-SCORES ─────────────────────────────────────────────────────────
        {_format_maturity_overview(scores, descriptions)}

          Gesamtbereitschaft (Minimum aller Dimensionen): {overall:.1f} / 3

        ── BOTTLENECKS ──────────────────────────────────────────────────────────────
        {bottleneck_blocks}

        ── AUFGABE ──────────────────────────────────────────────────────────────────
        Erstelle eine klare, priorisierte Handlungsempfehlung für das Unternehmen \
        "{assessment.company_id}" (Branche: {industry_str}) auf Basis der oben stehenden Daten.
        Folge dabei der vorgegebenen Struktur aus deinem System-Prompt.
        
        WICHTIG: Gib bei JEDER Empfehlung konkrete, branchenspezifische Hinweise:
        - WENN Technologie-Änderungen empfohlen werden → nenne zugehörige Schulungs- und Akzeptanz-Maßnahmen
        - WENN Prozess-Änderungen empfohlen werden → nenne die verantwortliche Rolle und typische Branchen-Fallstricke
        - WENN Akzeptanz-Probleme sichtbar sind → empfiehl konkrete Change-Kommunikations-Maßnahmen für diese Branche
        
        Beziche dich auf das Branchen-Know-how aus deiner Systemanweisung und gib konkrete Beispiele 
        für typische Maßnahmen in der Branche {industry_str}.
        """.strip()

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
