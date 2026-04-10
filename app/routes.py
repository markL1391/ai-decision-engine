"""
routes.py

Flask routes for the Explainable AI Maturity Assessment System.

This module provides:
- HTML demo routes for the frontend
- API endpoints for deterministic assessment analysis
- AI explanation generation with conversation history retention
- comparative analysis endpoints
- health check endpoint

Project requirements covered here:
- Flask API with multiple GET and POST endpoints
- SQLite DB insert and read flows
- one text generation endpoint that updates the DB
- structured outputs from the LLM layer
- use-case-specific comparative analysis
- prompt engineering via constrained prompting + contextual injection
- retaining conversation history
"""

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List

from flask import Blueprint, jsonify, render_template, request, session
from pydantic import ValidationError

from app import db
from app.models import (
    Assessment,
    ConversationTurn,
    Explanation,
    IndicatorScore,
    Metric,
    Result,
)
from app.schemas import (
    AssessmentAnalyzeRequest,
    CompareAssessmentsRequest,
    ExplanationGenerateRequest,
)
from app.llm import (
    build_compare_prompt,
    build_prompt,
    generate_compare_explanation_openai,
    generate_explanation_openai,
    generate_explanation_openai_with_temperature,
    generate_temperature_comparison_openai
)
from core.engine import run_deterministic_engine
from core.mapping import map_metrics_to_indicators
from core.rag import retrieve_context
from core.output import OutputParseError, parse_llm_output
from core.custom_mapping import map_custom_kpis_to_indicators
from core.ai_mapping import suggest_dimension_for_kpi

api_bp = Blueprint("api", __name__)

DIMENSION_DISPLAY = {
    "de": {
        "T": "Technologie",
        "P": "Prozess",
        "R": "Verantwortung",
        "A": "Akzeptanz",
    },
    "en": {
        "T": "Technology",
        "P": "Process",
        "R": "Responsibility",
        "A": "Adoption",
    }
}

def get_dimension_display(lang, key):
    return DIMENSION_DISPLAY.get(lang, DIMENSION_DISPLAY["en"]).get(key, key)

TEXTS = {
    "de": {
        "hero_badge": "Erklärbare KI für operative Systeme",
        "hero_title": "Explainable Decision Engine",
        "hero_tagline": "Operative Kennzahlen in strukturelle Einsichten, sichtbare Engpässe und entscheidungsreifes Systemverständnis übersetzen.",
        "hero_description": "Das System bewertet aktuelle Reifegrade, vergleicht Zukunftsszenarien, benchmarkt Unternehmen auf einer gemeinsamen Struktur und erlaubt sogar unternehmensspezifische KPI-Mappings.",
        "mode_1_label": "Modus 1",
        "mode_1_title": "Ist-Zustand analysieren",
        "mode_1_text": "Bewerte den aktuellen Reifegrad deines Systems und identifiziere, welche strukturellen Engpässe die operative Stabilität begrenzen.",
        "mode_2_label": "Modus 2",
        "mode_2_title": "Szenarien vergleichen",
        "mode_2_text": "Simuliere ein Vorher-Nachher-Szenario und prüfe, ob eine geplante Maßnahme die eigentlichen Bottlenecks wirklich beseitigt.",
        "mode_3_label": "Modus 3",
        "mode_3_title": "Unternehmen benchmarken",
        "mode_3_text": "Vergleiche mehrere Unternehmen auf derselben strukturellen Logik und erkenne wiederkehrende Engpassprofile.",
        "mode_4_label": "Modus 4",
        "mode_4_title": "Custom KPI Mapping",
        "mode_4_text": "Nutze unternehmensspezifische Kennzahlen und übersetze sie in dieselben universellen Systemdimensionen.",
        "mode_5_label": "Modus 5",
        "mode_5_title": "AI Mapping Helper",
        "mode_5_text": "Lass das System für neue KPI-Namen eine erste strukturelle Zuordnung vorschlagen.",
        "input_metrics_title": "Eingabe Kennzahlen",
        "company_label": "Unternehmen",
        "company_placeholder": "z.B. Nordic Express GmbH",
        "industry_label": "Branche",
        "industry_placeholder": "z.B. Logistik, Produktion, E-Commerce",
        "industry_help": "Optional – verbessert die KI-Empfehlung",
        "automation_rate_label": "Automatisierungsgrad (%)",
        "automation_rate_help": "Anteil automatisierter Abläufe",
        "system_availability_label": "Systemverfügbarkeit (%)",
        "system_availability_help": "Systemverfügbarkeit im Betrieb",
        "error_rate_label": "Fehlerquote (%)",
        "error_rate_help": "Fehlerquote im Prozess",
        "order_processing_time_label": "Bearbeitungszeit (min)",
        "order_processing_time_help": "Durchschnittliche Bearbeitungszeit",
        "process_standardization_label": "Prozessstandardisierung",
        "process_standardization_help": "Wie standardisiert sind die Prozesse?",
        "role_clarity_label": "Rollenklarheit",
        "role_clarity_help": "Wie klar sind Rollen definiert?",
        "ownership_definition_label": "Verantwortungsdefinition",
        "ownership_definition_help": "Wie verbindlich ist Ownership geregelt?",
        "training_coverage_label": "Schulungsabdeckung (%)",
        "training_coverage_help": "Abdeckung von Trainingsmaßnahmen",
        "tool_adoption_label": "Tool-Nutzung (%)",
        "tool_adoption_help": "Nutzung neuer Tools im Alltag",
        "change_communication_label": "Change-Kommunikation",
        "change_communication_help": "Wie strukturiert wird Veränderung kommuniziert?",
        "test_data_button": "Testdaten laden",
        "analyze_button": "Analyse starten",
        "language_switch_de": "DE",
        "language_switch_en": "EN",
    },
    "en": {
        "hero_badge": "Explainable AI for operational systems",
        "hero_title": "Explainable Decision Engine",
        "hero_tagline": "Turn operational metrics into structural insight, visible bottlenecks, and decision-ready system understanding.",
        "hero_description": "The system assesses current maturity, compares future scenarios, benchmarks companies on a shared structure, and even supports company-specific KPI mapping.",
        "mode_1_label": "Mode 1",
        "mode_1_title": "Analyze current state",
        "mode_1_text": "Assess the current maturity of your system and identify which structural bottlenecks limit operational stability.",
        "mode_2_label": "Mode 2",
        "mode_2_title": "Compare scenarios",
        "mode_2_text": "Simulate a before-and-after scenario and test whether a planned change actually removes the real bottlenecks.",
        "mode_3_label": "Mode 3",
        "mode_3_title": "Benchmark companies",
        "mode_3_text": "Compare multiple companies on the same structural logic and identify recurring bottleneck patterns.",
        "mode_4_label": "Mode 4",
        "mode_4_title": "Custom KPI mapping",
        "mode_4_text": "Use company-specific KPIs and translate them into the same universal system dimensions.",
        "mode_5_label": "Mode 5",
        "mode_5_title": "AI mapping helper",
        "mode_5_text": "Let the system suggest an initial structural mapping for new KPI names.",
        "input_metrics_title": "Input Metrics",
        "company_label": "Company",
        "company_placeholder": "e.g. Nordic Express GmbH",
        "industry_label": "Industry",
        "industry_placeholder": "e.g. Logistics, Manufacturing, E-Commerce",
        "industry_help": "Optional – improves AI recommendations",
        "automation_rate_label": "Automation Rate (%)",
        "automation_rate_help": "Share of automated processes",
        "system_availability_label": "System Availability (%)",
        "system_availability_help": "System uptime in operation",
        "error_rate_label": "Error Rate (%)",
        "error_rate_help": "Error rate in the process",
        "order_processing_time_label": "Processing Time (min)",
        "order_processing_time_help": "Average processing time",
        "process_standardization_label": "Process Standardization",
        "process_standardization_help": "How standardized are the processes?",
        "role_clarity_label": "Role Clarity",
        "role_clarity_help": "How clearly are roles defined?",
        "ownership_definition_label": "Ownership Definition",
        "ownership_definition_help": "How binding is ownership regulated?",
        "training_coverage_label": "Training Coverage (%)",
        "training_coverage_help": "Coverage of training measures",
        "tool_adoption_label": "Tool Adoption (%)",
        "tool_adoption_help": "Usage of new tools in daily work",
        "change_communication_label": "Change Communication",
        "change_communication_help": "How structured is change communicated?",
        "test_data_button": "Load Sample Data",
        "analyze_button": "Start Analysis",
        "language_switch_de": "DE",
        "language_switch_en": "EN",
    }
}

# =============================================================================
# Helper functions
# =============================================================================

def _error_response(message: str, status_code: int, details: Any = None):
    """
    Return a consistent JSON error response.

    Args:
        message: Human-readable error message.
        status_code: HTTP status code.
        details: Optional additional error details.

    Returns:
        Flask JSON response tuple.
    """
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status_code


def _translate_to_german(text: str, api_key: str) -> str:
    """
    Translate English text to German using OpenAI.

    Args:
        text: English text to translate.
        api_key: OpenAI API key.

    Returns:
        German translation of the text.
    """
    if not text or not text.strip():
        return text

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[{
                "role": "user",
                "content": f"Translate the following text to German. Keep the tone professional and consistent with business/consulting language. Do not add explanations, only the translation.\n\nText:\n{text}"
            }],
        )

        return response.output_text.strip()
    except Exception:
        return text


def _translate_structured_output(output: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Translate a structured advisory output from English to German.

    Args:
        output: Dictionary with summary, priorities, lever, risk, next_step, etc.
        api_key: OpenAI API key.

    Returns:
        Translated dictionary.
    """
    translated = {}

    if output.get("summary"):
        translated["summary"] = _translate_to_german(output["summary"], api_key)
    else:
        translated["summary"] = output.get("summary", "")

    if output.get("top_priorities"):
        translated_priorities = []
        for p in output["top_priorities"]:
            tp = dict(p)
            if p.get("action"):
                tp["action"] = _translate_to_german(p["action"], api_key)
            if p.get("rationale"):
                tp["rationale"] = _translate_to_german(p["rationale"], api_key)
            if p.get("timeframe"):
                tp["timeframe"] = _translate_to_german(p["timeframe"], api_key)
            translated_priorities.append(tp)
        translated["top_priorities"] = translated_priorities
    else:
        translated["top_priorities"] = output.get("top_priorities", [])

    if output.get("lever"):
        tl = dict(output["lever"])
        if tl.get("explanation"):
            tl["explanation"] = _translate_to_german(tl["explanation"], api_key)
        translated["lever"] = tl
    else:
        translated["lever"] = output.get("lever", {})

    if output.get("risk"):
        tr = dict(output["risk"])
        if tr.get("consequence"):
            tr["consequence"] = _translate_to_german(tr["consequence"], api_key)
        translated["risk"] = tr
    else:
        translated["risk"] = output.get("risk", {})

    if output.get("next_step"):
        tns = dict(output["next_step"])
        if tns.get("action"):
            tns["action"] = _translate_to_german(tns["action"], api_key)
        if tns.get("owner"):
            tns["owner"] = _translate_to_german(tns["owner"], api_key)
        if tns.get("by_when"):
            tns["by_when"] = _translate_to_german(tns["by_when"], api_key)
        translated["next_step"] = tns
    else:
        translated["next_step"] = output.get("next_step", {})

    translated["rag_references"] = output.get("rag_references", [])

    return translated


def _build_demo_metrics(form_data) -> List[SimpleNamespace]:
    """
    Parse HTML form inputs into Metric-like objects for the demo route.

    The deterministic mapping layer expects objects with .name and .value.
    SimpleNamespace is sufficient for that purpose.

    Args:
        form_data: Flask request.form object.

    Returns:
        List of SimpleNamespace metrics.
    """
    return [
        SimpleNamespace(name="automation_rate", value=float(form_data.get("automation_rate"))),
        SimpleNamespace(name="system_availability", value=float(form_data.get("system_availability"))),
        SimpleNamespace(name="error_rate", value=float(form_data.get("error_rate"))),
        SimpleNamespace(name="order_processing_time", value=float(form_data.get("order_processing_time"))),
        SimpleNamespace(name="process_standardization", value=form_data.get("process_standardization")),
        SimpleNamespace(name="role_clarity", value=form_data.get("role_clarity")),
        SimpleNamespace(name="ownership_definition", value=form_data.get("ownership_definition")),
        SimpleNamespace(name="training_coverage", value=float(form_data.get("training_coverage"))),
        SimpleNamespace(name="tool_adoption", value=float(form_data.get("tool_adoption"))),
        SimpleNamespace(name="change_communication", value=form_data.get("change_communication")),
    ]


def _build_comparison_payload(
    result_a: Dict[str, Any], 
    result_b: Dict[str, Any],
    company_name: str = "",
    industry: str = ""
) -> Dict[str, Any]:
    """
    Build a comparison dictionary from two deterministic assessment results.

    Args:
        result_a: Deterministic result of assessment A.
        result_b: Deterministic result of assessment B.
        company_name: Optional company name for context.
        industry: Optional industry for branch-specific recommendations.

    Returns:
        Dictionary containing score deltas and feasibility comparison.
    """
    t_delta = round(result_b["dimension_scores"].get("T", 0) - result_a["dimension_scores"].get("T", 0), 2)
    a_delta = round(result_b["dimension_scores"].get("A", 0) - result_a["dimension_scores"].get("A", 0), 2)

    risk_alert_status = False
    if t_delta >= 1.0 and a_delta <= 0.2:
        risk_alert_status = True

    critical_gap_value = round(t_delta - a_delta, 2)

    return {
        "company_name": company_name or "the company",
        "industry": industry or "",
        "r_delta": round(result_b["dimension_scores"].get("R", 0) - result_a["dimension_scores"].get("R", 0), 2),
        "p_delta": round(result_b["dimension_scores"].get("P", 0) - result_a["dimension_scores"].get("P", 0), 2),
        "t_delta": round(result_b["dimension_scores"].get("T", 0) - result_a["dimension_scores"].get("T", 0), 2),
        "a_delta": round(result_b["dimension_scores"].get("A", 0) - result_a["dimension_scores"].get("A", 0), 2),
        "overall_readiness_delta": round(result_b["overall_readiness"] - result_a["overall_readiness"], 2),
        "bottleneck_a": result_a["bottlenecks"],
        "bottleneck_b": result_b["bottlenecks"],
        "transition_feasible_a": result_a["transition_feasible"],
        "transition_feasible_b": result_b["transition_feasible"],
        "risk_alert": risk_alert_status,
        "critical_gap": critical_gap_value,
    }

def _build_comparison_from_db(a: Assessment, b: Assessment) -> Dict[str, Any]:
    """
    Build a comparison dictionary using stored DB results.

    Args:
        a: Assessment A from the database.
        b: Assessment B from the database.

    Returns:
        Dictionary containing deltas and feasibility comparison.
    """
    return {
        "company_name": a.metrics.company_id if hasattr(a.metrics, 'company_id') else "Company",
        "industry": a.metrics.industry if hasattr(a.metrics, 'industry') else "",
        "r_delta": round(b.result.r_score - a.result.r_score, 2),
        "p_delta": round(b.result.p_score - a.result.p_score, 2),
        "t_delta": round(b.result.t_score - a.result.t_score, 2),
        "a_delta": round(b.result.a_score - a.result.a_score, 2),
        "overall_readiness_delta": round(b.result.overall_readiness - a.result.overall_readiness, 2),
        "bottleneck_a": json.loads(a.result.bottlenecks_json),
        "bottleneck_b": json.loads(b.result.bottlenecks_json),
        "transition_feasible_a": a.result.transition_feasible,
        "transition_feasible_b": b.result.transition_feasible,
    }


def _build_demo_summary(engine_result: Dict[str, Any]) -> str:
    """
    Build a simple deterministic summary for the demo route.

    Args:
        engine_result: Output of the deterministic engine.

    Returns:
        Short summary string.
    """
    if engine_result["transition_feasible"]:
        return (
            "The system currently meets the target level and no structural bottleneck "
            "is blocking the transition."
        )

    if engine_result["bottlenecks"]:
        joined = ", ".join(engine_result["bottlenecks"])
        return (
            f"The system is currently not transition-ready because the dimensions "
            f"{joined} remain below the required target level."
        )

    return "The system has been analysed successfully."


def _serialize_assessment(assessment: Assessment) -> Dict[str, Any]:
    """
    Serialize a full assessment record including metrics, indicator scores, and results.

    Args:
        assessment: Assessment SQLAlchemy object.

    Returns:
        Dictionary suitable for JSON response.
    """
    metrics = [
        {
            "name": m.name,
            "value": m.value,
            "unit": m.unit,
        }
        for m in assessment.metrics
    ]

    indicators = [
        {
            "dimension": i.dimension,
            "indicator": i.indicator,
            "value": i.value,
        }
        for i in assessment.indicator_scores
    ]

    result = None
    if assessment.result:
        result = {
            "r_score": assessment.result.r_score,
            "p_score": assessment.result.p_score,
            "t_score": assessment.result.t_score,
            "a_score": assessment.result.a_score,
            "overall_readiness": assessment.result.overall_readiness,
            "bottlenecks": json.loads(assessment.result.bottlenecks_json),
            "transition_feasible": assessment.result.transition_feasible,
            "transition_risk": assessment.result.transition_risk,
            "required_changes": json.loads(assessment.result.required_changes_json),
        }

    explanation = None
    if assessment.explanation:
        explanation = {
            "why_limit": json.loads(assessment.explanation.why_limit_json),
            "blocks_transition": json.loads(assessment.explanation.blocks_transition_json),
            "references": json.loads(assessment.explanation.references_json)
            if assessment.explanation.references_json else [],
            "model_name": assessment.explanation.model_name,
            "prompt_version": assessment.explanation.prompt_version,
        }

    conversation_history = [
        {
            "id": turn.id,
            "role": turn.role,
            "content": turn.content,
            "created_at": turn.created_at.isoformat(),
        }
        for turn in assessment.conversation_turns
    ]

    return {
        "id": assessment.id,
        "created_at": assessment.created_at.isoformat(),
        "domain": assessment.domain,
        "notes": assessment.notes,
        "target_level": assessment.target_level,
        "metrics": metrics,
        "indicator_scores": indicators,
        "results": result,
        "explanation": explanation,
        "conversation_history": conversation_history,
    }


def _store_conversation_turn(assessment_id: int, role: str, content: str) -> None:
    """
    Persist a single conversation turn in the database.

    Args:
        assessment_id: Assessment ID the message belongs to.
        role: Usually 'user', 'assistant', or 'system'.
        content: Prompt, explanation text, or structured JSON string.
    """
    turn = ConversationTurn(
        assessment_id=assessment_id,
        role=role,
        content=content,
    )
    db.session.add(turn)


def _load_recent_history(assessment_id: int, limit: int = 5) -> List[ConversationTurn]:
    """
    Load recent conversation turns for one assessment.

    Args:
        assessment_id: Assessment ID.
        limit: Maximum number of turns to load.

    Returns:
        List of conversation turns ordered from oldest to newest within the selected window.
    """
    turns = (
        ConversationTurn.query
        .filter_by(assessment_id=assessment_id)
        .order_by(ConversationTurn.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(turns))


def _history_to_context_lines(turns: List[ConversationTurn]) -> List[str]:
    """
    Convert conversation history into short textual context items.

    These lines are injected into the retrieved context so that the LLM
    sees both domain context and prior interaction context.

    Args:
        turns: List of ConversationTurn objects.

    Returns:
        List of textual context strings.
    """
    lines: List[str] = []

    for turn in turns:
        if turn.role == "user":
            lines.append(f"Previous user request: {turn.content}")
        elif turn.role == "assistant":
            lines.append(f"Previous assistant response: {turn.content}")
        else:
            lines.append(f"Previous system context: {turn.content}")

    return lines


def _save_assessment_and_result(payload: AssessmentAnalyzeRequest) -> Dict[str, Any]:
    """
    Persist an assessment, its raw metrics, mapped indicators, and deterministic result.

    Args:
        payload: Validated assessment request payload.

    Returns:
        Dictionary with assessment and engine result.
    """
    assessment = Assessment(
        domain=payload.assessment.domain,
        notes=payload.assessment.notes,
        target_level=payload.assessment.target_level,
    )
    db.session.add(assessment)
    db.session.flush()

    for metric in payload.metrics:
        db.session.add(
            Metric(
                assessment_id=assessment.id,
                name=metric.name,
                value=str(metric.value),
                unit=metric.unit,
            )
        )

    mapped_indicators = map_metrics_to_indicators(payload.metrics)
    engine_result = run_deterministic_engine(
        mapped_indicators,
        payload.assessment.target_level,
    )

    for item in mapped_indicators:
        db.session.add(
            IndicatorScore(
                assessment_id=assessment.id,
                dimension=item["dimension"],
                indicator=item["indicator"],
                value=item["value"],
            )
        )

    db.session.add(
        Result(
            assessment_id=assessment.id,
            r_score=engine_result["dimension_scores"].get("R", 0),
            p_score=engine_result["dimension_scores"].get("P", 0),
            t_score=engine_result["dimension_scores"].get("T", 0),
            a_score=engine_result["dimension_scores"].get("A", 0),
            overall_readiness=engine_result["overall_readiness"],
            bottlenecks_json=json.dumps(engine_result["bottlenecks"]),
            transition_feasible=engine_result["transition_feasible"],
            transition_risk=engine_result["transition_risk"],
            required_changes_json=json.dumps(engine_result["required_changes"]),
        )
    )

    db.session.commit()

    return {
        "assessment": assessment,
        "engine_result": engine_result,
    }

def _derive_reference_labels(retrieved_context: List[str]) -> List[str]:
    """
    Derive simple frontend-friendly reference labels from retrieved context.

    This avoids overly artificial or generic LLM-generated source names.
    """
    labels: List[str] = []

    joined = " ".join(retrieved_context).lower()

    if "technology" in joined:
        labels.append("Technology bottleneck context")
    if "process" in joined:
        labels.append("Process bottleneck context")
    if "responsibility" in joined or "ownership" in joined:
        labels.append("Responsibility bottleneck context")
    if "adoption" in joined or "acceptance" in joined:
        labels.append("Adoption bottleneck context")

    if not labels and retrieved_context:
        labels.append("Assessment context")

    return labels

def _build_demo_metrics_with_prefix(form_data, prefix: str) -> List[SimpleNamespace]:
    """
    Parse one prefixed HTML form section into Metric-like objects.

    Example prefix:
    - "a_" for scenario A
    - "b_" for scenario B
    """
    return [
        SimpleNamespace(name="automation_rate", value=float(form_data.get(f"{prefix}automation_rate"))),
        SimpleNamespace(name="system_availability", value=float(form_data.get(f"{prefix}system_availability"))),
        SimpleNamespace(name="error_rate", value=float(form_data.get(f"{prefix}error_rate"))),
        SimpleNamespace(name="order_processing_time", value=float(form_data.get(f"{prefix}order_processing_time"))),
        SimpleNamespace(name="process_standardization", value=form_data.get(f"{prefix}process_standardization")),
        SimpleNamespace(name="role_clarity", value=form_data.get(f"{prefix}role_clarity")),
        SimpleNamespace(name="ownership_definition", value=form_data.get(f"{prefix}ownership_definition")),
        SimpleNamespace(name="training_coverage", value=float(form_data.get(f"{prefix}training_coverage"))),
        SimpleNamespace(name="tool_adoption", value=float(form_data.get(f"{prefix}tool_adoption"))),
        SimpleNamespace(name="change_communication", value=form_data.get(f"{prefix}change_communication")),
    ]

def _get_language() -> str:
    """
    Get the UI language from the query parameter.
    Defaults to German.
    """
    lang = request.args.get("lang", "de").lower()
    return "en" if lang == "en" else "de"

def _get_session_chat_history() -> List[Dict[str, str]]:
    """
    Return the current session-based chat history.
    """
    return session.get("chat_history", [])


def _save_session_chat_history(history: List[Dict[str, str]]) -> None:
    """
    Persist chat history in the Flask session.
    """
    session["chat_history"] = history
    session.modified = True


def _append_session_message(role: str, content: str) -> None:
    """
    Append one chat message to the session history.
    """
    history = _get_session_chat_history()
    history.append({"role": role, "content": content})
    _save_session_chat_history(history)


def _clear_session_chat_history() -> None:
    """
    Clear chat history in the current session.
    """
    session.pop("chat_history", None)
    session.modified = True

# =============================================================================
# Frontend / demo routes
# =============================================================================
@api_bp.route("/landing")
def landing():
    lang = _get_language()
    return render_template(
        "landing.html",
        lang=lang,
    )

@api_bp.route("/", methods=["GET"])
def home():
    """
    Render the main HTML frontend.

    This route exists mainly for demo and presentation purposes.
    """
    lang = _get_language()
    texts = TEXTS[lang]
    return render_template(
        "index.html",
        lang=lang,
        texts=texts,
        result=None,
        assessment=None,
        summary=None,
        priorities=[],
        lever=None,
        risk=None,
        next_step=None,
        references=[],
        input_metrics=None,
        indicator_view=[],
    )

@api_bp.route("/demo", methods=["POST"])
def demo():
    """
    Run the end-to-end demo flow from an HTML form submission.

    Flow:
    1. Parse form metrics
    2. Map metrics to indicators
    3. Run deterministic engine
    4. Retrieve contextual knowledge
    5. Generate structured AI advisory output
    6. Validate via output.py
    7. Render the frontend with all expected fields
    """

    lang = _get_language()
    texts = TEXTS[lang]
    metrics = _build_demo_metrics(request.form)
    mapped_indicators = map_metrics_to_indicators(metrics)
    
    # Add custom KPIs if provided
    custom_kpis = None
    custom_kpis_json = request.form.get("custom_kpis_json")
    if custom_kpis_json:
        try:
            import json
            custom_kpis = json.loads(custom_kpis_json)
            if custom_kpis and len(custom_kpis) > 0:
                custom_indicators = map_custom_kpis_to_indicators(custom_kpis)
                mapped_indicators.extend(custom_indicators)
        except Exception:
            custom_kpis = None
    
    result = run_deterministic_engine(mapped_indicators, target_level=2)
    indicator_view = mapped_indicators

    summary = _build_demo_summary(result)
    priorities = []
    lever = None
    risk = None
    next_step = None
    references = []

    input_metrics = {
        "automation_rate": float(request.form.get("automation_rate")),
        "system_availability": float(request.form.get("system_availability")),
        "error_rate": float(request.form.get("error_rate")),
        "order_processing_time": float(request.form.get("order_processing_time")),
        "process_standardization": request.form.get("process_standardization"),
        "role_clarity": request.form.get("role_clarity"),
        "ownership_definition": request.form.get("ownership_definition"),
        "training_coverage": float(request.form.get("training_coverage")),
        "tool_adoption": float(request.form.get("tool_adoption")),
        "change_communication": request.form.get("change_communication"),
    }

    session["engine_result"] = result
    session["assessment_context"] = {
        "company_id": request.form.get("company_id", "Demo Company"),
        "industry": request.form.get("industry", ""),
    }
    session["last_summary"] = summary
    session.modified = True

    assessment = SimpleNamespace(
        company_id=request.form.get("company_id", "Demo Company"),
        industry=request.form.get("industry"),
        target_level=2,
    )

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            retrieved_context = retrieve_context(result)

            raw_output = generate_explanation_openai(
                api_key=api_key,
                engine_result=result,
                retrieved_context=retrieved_context,
                industry=assessment.industry or "",
            )

            structured = parse_llm_output(json.dumps({
                "summary": raw_output["summary"],
                "top_priorities": raw_output["top_priorities"],
                "lever": raw_output["lever"],
                "risk": raw_output["risk"],
                "next_step": raw_output["next_step"],
                "rag_references": raw_output["rag_references"],
            }))

            raw_structured = {
                "summary": structured.summary,
                "top_priorities": [p.model_dump() for p in structured.top_priorities],
                "lever": structured.lever.model_dump(),
                "risk": structured.risk.model_dump(),
                "next_step": structured.next_step.model_dump(),
                "rag_references": raw_output.get("rag_references", []),
            }

            if lang == "de":
                raw_structured = _translate_structured_output(raw_structured, api_key)

            summary = raw_structured["summary"]

            priorities = []
            for p in raw_structured["top_priorities"]:
                item = dict(p)
                item["dimension_label"] = get_dimension_display(lang, item["dimension"])
                priorities.append(item)

            lever = dict(raw_structured["lever"])
            lever["dimension_label"] = get_dimension_display(lang, lever["dimension"])

            risk = dict(raw_structured["risk"])
            next_step = dict(raw_structured["next_step"])

            references = _derive_reference_labels(retrieved_context)

        except OutputParseError as e:
            summary = f"The system was analysed, but structured output validation failed: {e}"
        except Exception as e:
            summary = f"The system was analysed, but the AI summary could not be generated: {e}"

    return render_template(
        "index.html",
        result=result,
        assessment=assessment,
        summary=summary,
        priorities=priorities,
        lever=lever,
        risk=risk,
        next_step=next_step,
        references=references,
        input_metrics=input_metrics,
        indicator_view=indicator_view,
        custom_kpis=custom_kpis,
        lang=lang,
        texts=texts,
    )

@api_bp.route("/compare-demo", methods=["GET", "POST"])
def compare_demo():
    """
    Minimal compare demo without database persistence.

    This route allows users to compare:
    - Scenario A = current state
    - Scenario B = future / implementation scenario

    Flow:
    1. Parse both form sections
    2. Run deterministic engine for A and B
    3. Build comparison payload
    4. Optionally generate AI comparison explanation
    5. Render compare demo template
    """
    import logging
    logger = logging.getLogger(__name__)
    
    lang = _get_language()
    texts = TEXTS[lang]
    template_context = {
        "scenario_a": None,
        "scenario_b": None,
        "result_a": None,
        "result_b": None,
        "comparison": None,
        "compare_explanation": None,
        "error": None,
    }

    if request.method == "GET":
        logger.info("Compare demo GET request")
        return render_template("compare_demo.html", lang=lang, texts=texts, **template_context)

    try:
        logger.info("Compare demo POST request received")
        logger.info(f"Form fields: {list(request.form.keys())}")
        
        metrics_a = _build_demo_metrics_with_prefix(request.form, "a_")
        metrics_b = _build_demo_metrics_with_prefix(request.form, "b_")
        logger.info(f"Metrics A: {[m.name for m in metrics_a]}")
        logger.info(f"Metrics B: {[m.name for m in metrics_b]}")

        mapped_a = map_metrics_to_indicators(metrics_a)
        mapped_b = map_metrics_to_indicators(metrics_b)
        logger.info(f"Mapped A: {mapped_a}")
        logger.info(f"Mapped B: {mapped_b}")

        result_a = run_deterministic_engine(mapped_a, target_level=2)
        result_b = run_deterministic_engine(mapped_b, target_level=2)
        logger.info(f"Result A: {result_a.get('dimension_scores')}")
        logger.info(f"Result B: {result_b.get('dimension_scores')}")

        comparison = _build_comparison_payload(
            result_a, 
            result_b,
            company_name=request.form.get("a_company_id", "Scenario A"),
            industry=request.form.get("a_industry", "")
        )
        logger.info(f"Comparison built: t_delta={comparison.get('t_delta')}, p_delta={comparison.get('p_delta')}")

        compare_explanation = None
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                logger.info("Generating AI comparison explanation...")
                compare_explanation = generate_compare_explanation_openai(
                    api_key=api_key,
                    comparison=comparison,
                )
                logger.info("AI explanation generated successfully")

                if lang == "de" and compare_explanation:
                    if compare_explanation.get("summary"):
                        compare_explanation["summary"] = _translate_to_german(compare_explanation["summary"], api_key)
                    if compare_explanation.get("main_improvements"):
                        compare_explanation["main_improvements"] = [
                            _translate_to_german(item, api_key) for item in compare_explanation["main_improvements"]
                        ]
                    if compare_explanation.get("transition_impact"):
                        compare_explanation["transition_impact"] = [
                            _translate_to_german(item, api_key) for item in compare_explanation["transition_impact"]
                        ]
            except Exception as e:
                logger.error(f"AI explanation failed: {e}")
                compare_explanation = {
                    "summary": "The comparison explanation could not be generated.",
                    "main_improvements": [str(e)],
                    "transition_impact": [],
                }
        else:
            logger.warning("OPENAI_API_KEY not configured, skipping AI explanation")

        scenario_a = SimpleNamespace(
            company_id=request.form.get("a_company_id", "Scenario A"),
            industry=request.form.get("a_industry"),
        )

        scenario_b = SimpleNamespace(
            company_id=request.form.get("b_company_id", "Scenario B"),
            industry=request.form.get("b_industry"),
        )

        template_context.update(
            {
                "scenario_a": scenario_a,
                "scenario_b": scenario_b,
                "result_a": result_a,
                "result_b": result_b,
                "comparison": comparison,
                "compare_explanation": compare_explanation,
            }
        )
        logger.info("Rendering template with comparison data")

        return render_template(
            "compare_demo.html",
            lang=lang,
            texts=texts,
            **template_context
        )


    except Exception as e:
        logger.error(f"Compare demo exception: {e}", exc_info=True)
        template_context["error"] = f"Compare demo failed: {e}"
        return render_template("compare_demo.html", lang=lang, texts=texts, **template_context)

# =============================================================================
# API routes
# =============================================================================

@api_bp.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    This is one of the required GET endpoints and can be used to verify that
    the API is reachable and responsive.
    """
    return jsonify({"status": "ok"}), 200


@api_bp.route("/assessments/analyze", methods=["POST"])
def analyze_assessment():
    """
    Create and analyze an assessment.

    This endpoint:
    1. validates the incoming payload with Pydantic
    2. stores the assessment and raw metrics in SQLite
    3. maps raw inputs to normalized indicators
    4. runs the deterministic engine
    5. stores the result in the DB
    6. returns the structured assessment result
    """
    data = request.get_json(silent=True)
    if not data:
        return _error_response("Invalid or missing JSON body", 400)

    try:
        payload = AssessmentAnalyzeRequest(**data)
    except ValidationError as e:
        return _error_response("Validation failed", 422, e.errors())

    try:
        saved = _save_assessment_and_result(payload)
    except Exception as e:
        db.session.rollback()
        return _error_response("Assessment could not be saved", 500, str(e))

    assessment = saved["assessment"]
    engine_result = saved["engine_result"]
    session["assessment_id"] = assessment.id
    session.modified = True

    return jsonify({
        "message": "Assessment created and analyzed successfully",
        "assessment_id": assessment.id,
        "engine_result": {
            "dimension_scores": engine_result["dimension_scores"],
            "overall_readiness": engine_result["overall_readiness"],
            "bottlenecks": engine_result["bottlenecks"],
            "required_changes": engine_result["required_changes"],
            "required_capacities": engine_result["required_capacities"],
            "bottleneck_details": engine_result["bottleneck_details"],
            "transition_feasible": engine_result["transition_feasible"],
            "transition_risk": engine_result["transition_risk"],
            "strengths": engine_result["strengths"],
            "weaknesses": engine_result["weaknesses"],
            "cross_dimension_insights": engine_result["cross_dimension_insights"],
            "executive_summary": engine_result["executive_summary"],
            "leverage_explanation": engine_result["leverage_explanation"],
        },
    }), 201


@api_bp.route("/assessments/<int:assessment_id>", methods=["GET"])
def get_assessment(assessment_id: int):
    """
    Read one assessment and all linked data from the database.

    This is one of the required GET endpoints and demonstrates DB read access.
    """
    assessment = Assessment.query.get(assessment_id)
    if not assessment:
        return _error_response("Assessment not found", 404)

    return jsonify(_serialize_assessment(assessment)), 200


@api_bp.route("/explanations/generate", methods=["POST"])
def generate_explanation():
    """
    Generate an AI explanation for an existing assessment and update the DB.
    """
    data = request.get_json(silent=True)
    if not data:
        return _error_response("Invalid or missing JSON body", 400)

    try:
        payload = ExplanationGenerateRequest(**data)
    except ValidationError as e:
        return _error_response("Validation failed", 422, e.errors())

    assessment = Assessment.query.get(payload.assessment_id)
    if not assessment:
        return _error_response("Assessment not found", 404)

    if not assessment.result:
        return _error_response("No deterministic result found for assessment", 404)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _error_response("OPENAI_API_KEY is not configured.", 500)

    try:
        mapped_indicators = map_metrics_to_indicators(assessment.metrics)
        engine_result = run_deterministic_engine(mapped_indicators, assessment.target_level)

        retrieved_context = retrieve_context(engine_result)
        history_turns = _load_recent_history(assessment.id, limit=5)
        history_context = _history_to_context_lines(history_turns)
        combined_context = retrieved_context + history_context

        prompt_preview = build_prompt(engine_result, combined_context)

        _store_conversation_turn(
            assessment_id=assessment.id,
            role="user",
            content=prompt_preview,
        )

        raw_output = generate_explanation_openai(
            api_key=api_key,
            engine_result=engine_result,
            retrieved_context=combined_context,
        )

        structured = parse_llm_output(json.dumps({
            "summary": raw_output["summary"],
            "top_priorities": raw_output["top_priorities"],
            "lever": raw_output["lever"],
            "risk": raw_output["risk"],
            "next_step": raw_output["next_step"],
            "rag_references": raw_output["rag_references"],
        }))

        _store_conversation_turn(
            assessment_id=assessment.id,
            role="assistant",
            content=json.dumps(structured.to_dict()),
        )

        existing = Explanation.query.filter_by(assessment_id=assessment.id).first()

        priority_items = []
        for p in structured.top_priorities:
            item = p.model_dump()
            item["dimension_label"] = get_dimension_display("en", item["dimension"])
            priority_items.append(item)

        lever_item = structured.lever.model_dump()
        lever_item["dimension_label"] = get_dimension_display("en", lever_item["dimension"])

        explanation_payload = {
            "summary": structured.summary,
            "top_priorities": priority_items,
            "lever": lever_item,
            "risk": structured.risk.model_dump(),
            "next_step": structured.next_step.model_dump(),
            "rag_references": _derive_reference_labels(combined_context),
        }

        if existing:
            existing.why_limit_json = json.dumps(explanation_payload["summary"])
            existing.blocks_transition_json = json.dumps(explanation_payload["top_priorities"])
            existing.references_json = json.dumps(explanation_payload["rag_references"])
            existing.model_name = raw_output.get("model_name")
            existing.prompt_version = raw_output.get("prompt_version")
        else:
            db.session.add(
                Explanation(
                    assessment_id=assessment.id,
                    why_limit_json=json.dumps(explanation_payload["summary"]),
                    blocks_transition_json=json.dumps(explanation_payload["top_priorities"]),
                    references_json=json.dumps(explanation_payload["rag_references"]),
                    model_name=raw_output.get("model_name"),
                    prompt_version=raw_output.get("prompt_version"),
                )
            )

        db.session.commit()

    except OutputParseError as e:
        db.session.rollback()
        return _error_response("Structured output validation failed", 500, str(e))
    except Exception as e:
        db.session.rollback()
        return _error_response("OpenAI generation failed", 500, str(e))

    return jsonify({
        "assessment_id": assessment.id,
        "prompt_preview": prompt_preview,
        "conversation_history_used": [
            {
                "role": turn.role,
                "content": turn.content,
                "created_at": turn.created_at.isoformat(),
            }
            for turn in history_turns
        ],
        "engine_result": engine_result,
        "advisory_output": structured.to_dict(),
        "model_name": raw_output.get("model_name"),
        "prompt_version": raw_output.get("prompt_version"),
    }), 200

@api_bp.route("/assessments/compare", methods=["POST"])
def compare_assessments():
    """
    Compare two existing assessments based on stored deterministic results.

    This endpoint supports the project requirement for use-case-specific
    comparative analysis.
    """
    data = request.get_json(silent=True)
    if not data:
        return _error_response("Invalid or missing JSON body", 400)

    try:
        payload = CompareAssessmentsRequest(**data)
    except ValidationError as e:
        return _error_response("Validation failed", 422, e.errors())

    a = Assessment.query.get(payload.assessment_a_id)
    b = Assessment.query.get(payload.assessment_b_id)

    if not a or not b:
        return _error_response("One or both assessments not found", 404)

    if not a.result or not b.result:
        return _error_response("One or both assessments have no results.", 404)

    comparison = _build_comparison_from_db(a, b)

    return jsonify({
        "assessment_a_id": a.id,
        "assessment_b_id": b.id,
        "comparison": comparison,
    }), 200


@api_bp.route("/assessments/compare/explain", methods=["POST"])
def compare_and_explain():
    """
    Compare two assessments and generate an AI explanation of the structural differences.

    This endpoint combines:
    - deterministic comparison
    - prompt-based AI explanation
    - use-case-specific comparative analysis
    """
    data = request.get_json(silent=True)
    if not data:
        return _error_response("Invalid or missing JSON body", 400)

    try:
        payload = CompareAssessmentsRequest(**data)
    except ValidationError as e:
        return _error_response("Validation failed", 422, e.errors())

    a = Assessment.query.get(payload.assessment_a_id)
    b = Assessment.query.get(payload.assessment_b_id)

    if not a or not b:
        return _error_response("One or both assessments not found", 404)

    if not a.result or not b.result:
        return _error_response("One or both assessments have no results", 400)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _error_response("OPENAI_API_KEY is not configured.", 500)

    comparison = _build_comparison_from_db(a, b)
    prompt_preview = build_compare_prompt(comparison)

    try:
        explanation = generate_compare_explanation_openai(api_key, comparison)
    except Exception as e:
        return _error_response("OpenAI generation failed", 500, str(e))

    return jsonify({
        "assessment_a_id": a.id,
        "assessment_b_id": b.id,
        "comparison": comparison,
        "prompt_preview": prompt_preview,
        "explanation": explanation,
    }), 200

@api_bp.route("/benchmark-demo", methods=["GET"])
def benchmark_demo():
    """
    Demo route for multi-company benchmarking.

    This route runs a set of predefined company scenarios through the
    deterministic pipeline and renders a comparative benchmark view.
    """
    lang = _get_language()
    texts = TEXTS[lang]
    benchmark_cases = [
        {
            "name": "QuickDrop Logistics",
            "industry": "logistics",
            "metrics": [
                SimpleNamespace(name="automation_rate", value=32.0),
                SimpleNamespace(name="system_availability", value=92.0),
                SimpleNamespace(name="error_rate", value=8.5),
                SimpleNamespace(name="order_processing_time", value=24.0),
                SimpleNamespace(name="process_standardization", value="low"),
                SimpleNamespace(name="role_clarity", value="partial"),
                SimpleNamespace(name="ownership_definition", value="informal"),
                SimpleNamespace(name="training_coverage", value=45.0),
                SimpleNamespace(name="tool_adoption", value=52.0),
                SimpleNamespace(name="change_communication", value="irregular"),
            ],
        },
        {
            "name": "MediNow Pharmacy",
            "industry": "healthcare",
            "metrics": [
                SimpleNamespace(name="automation_rate", value=41.0),
                SimpleNamespace(name="system_availability", value=95.0),
                SimpleNamespace(name="error_rate", value=6.5),
                SimpleNamespace(name="order_processing_time", value=20.0),
                SimpleNamespace(name="process_standardization", value="medium"),
                SimpleNamespace(name="role_clarity", value="partial"),
                SimpleNamespace(name="ownership_definition", value="formal"),
                SimpleNamespace(name="training_coverage", value=62.0),
                SimpleNamespace(name="tool_adoption", value=58.0),
                SimpleNamespace(name="change_communication", value="structured"),
            ],
        },
        {
            "name": "FlowCart Commerce",
            "industry": "startup",
            "metrics": [
                SimpleNamespace(name="automation_rate", value=55.0),
                SimpleNamespace(name="system_availability", value=96.0),
                SimpleNamespace(name="error_rate", value=7.0),
                SimpleNamespace(name="order_processing_time", value=18.0),
                SimpleNamespace(name="process_standardization", value="medium"),
                SimpleNamespace(name="role_clarity", value="clear"),
                SimpleNamespace(name="ownership_definition", value="informal"),
                SimpleNamespace(name="training_coverage", value=71.0),
                SimpleNamespace(name="tool_adoption", value=78.0),
                SimpleNamespace(name="change_communication", value="structured"),
            ],
        },
        {
            "name": "Nordic Manufacturing Group",
            "industry": "manufacturing",
            "metrics": [
                SimpleNamespace(name="automation_rate", value=68.0),
                SimpleNamespace(name="system_availability", value=97.0),
                SimpleNamespace(name="error_rate", value=4.2),
                SimpleNamespace(name="order_processing_time", value=14.0),
                SimpleNamespace(name="process_standardization", value="high"),
                SimpleNamespace(name="role_clarity", value="clear"),
                SimpleNamespace(name="ownership_definition", value="formal"),
                SimpleNamespace(name="training_coverage", value=64.0),
                SimpleNamespace(name="tool_adoption", value=59.0),
                SimpleNamespace(name="change_communication", value="structured"),
            ],
        },
        {
            "name": "ScaleOS Tech Services",
            "industry": "tech",
            "metrics": [
                SimpleNamespace(name="automation_rate", value=74.0),
                SimpleNamespace(name="system_availability", value=98.0),
                SimpleNamespace(name="error_rate", value=3.0),
                SimpleNamespace(name="order_processing_time", value=12.0),
                SimpleNamespace(name="process_standardization", value="medium"),
                SimpleNamespace(name="role_clarity", value="clear"),
                SimpleNamespace(name="ownership_definition", value="formal"),
                SimpleNamespace(name="training_coverage", value=84.0),
                SimpleNamespace(name="tool_adoption", value=82.0),
                SimpleNamespace(name="change_communication", value="embedded"),
            ],
        },
    ]

    dimension_labels = {
        "T": "Technology",
        "P": "Process",
        "R": "Responsibility",
        "A": "Adoption",
    }

    benchmark_results = []

    for case in benchmark_cases:
        mapped = map_metrics_to_indicators(case["metrics"])
        result = run_deterministic_engine(mapped, target_level=2)

        scores = result["dimension_scores"]
        weakest_dim = min(scores, key=scores.get) if scores else None
        strongest_dim = max(scores, key=scores.get) if scores else None

        benchmark_results.append({
            "name": case["name"],
            "industry": case["industry"],
            "overall_readiness": result["overall_readiness"],
            "transition_feasible": result["transition_feasible"],
            "transition_risk": result["transition_risk"],
            "bottlenecks": result["bottlenecks"],
            "scores": scores,
            "weakest_dimension": dimension_labels.get(weakest_dim, weakest_dim),
            "strongest_dimension": dimension_labels.get(strongest_dim, strongest_dim),
        })

    benchmark_results.sort(key=lambda x: x["overall_readiness"], reverse=True)

    return render_template(
        "benchmark_demo.html",
        benchmark_results=benchmark_results,
        lang=lang,
        texts=texts,
    )

@api_bp.route("/custom-kpi-demo", methods=["GET", "POST"])
def custom_kpi_demo():
    """
    Demo route for user-defined KPI mapping.
    """
    lang = _get_language()
    texts = TEXTS[lang]
    if request.method == "GET":
        return render_template("custom_kpi_demo.html", lang=lang, texts=texts)

    try:
        custom_kpis = []

        names = request.form.getlist("kpi_name")
        values = request.form.getlist("kpi_value")
        directions = request.form.getlist("kpi_direction")
        dimensions = request.form.getlist("kpi_dimension")
        t1s = request.form.getlist("threshold_1")
        t2s = request.form.getlist("threshold_2")
        t3s = request.form.getlist("threshold_3")

        for i in range(len(names)):
            if not names[i].strip():
                continue

            custom_kpis.append({
                "name": names[i],
                "value": float(values[i]),
                "direction": directions[i],
                "dimension": dimensions[i],
                "thresholds": [float(t1s[i]), float(t2s[i]), float(t3s[i])],
            })

        indicators = map_custom_kpis_to_indicators(custom_kpis)
        result = run_deterministic_engine(indicators, target_level=2)

        return render_template(
            "custom_kpi_demo.html",
            custom_kpis=custom_kpis,
            indicators=indicators,
            result=result,
            lang=lang,
            texts=texts,
        )

    except Exception as e:
        return render_template(
            "custom_kpi_demo.html",
            error=f"Custom KPI demo failed: {e}",
            lang=lang,
            texts=texts,
        )

@api_bp.route("/ai-mapping-demo", methods=["GET", "POST"])
def ai_mapping_demo():
    """
    Demo route for simple AI-assisted KPI-to-dimension suggestions.
    """
    lang = _get_language()
    texts = TEXTS[lang]
    suggestion = None
    kpi_name_value = ""

    if request.method == "POST":
        kpi_name_value = request.form.get("kpi_name", "")
        if kpi_name_value.strip():
            suggestion = suggest_dimension_for_kpi(kpi_name_value, language=lang)
            
            if suggestion and "dimension_label" not in suggestion:
                labels = {
                    "de": {"T": "Technologie", "P": "Prozess", "R": "Verantwortung", "A": "Akzeptanz"},
                    "en": {"T": "Technology", "P": "Process", "R": "Responsibility", "A": "Acceptance"},
                }
                suggestion["dimension_label"] = labels.get(lang, labels["en"]).get(suggestion.get("dimension", "P"), "Prozess")

    return render_template(
        "ai_mapping_demo.html",
        suggestion=suggestion,
        kpi_name_value=kpi_name_value,
        lang=lang,
        texts=texts,
    )

@api_bp.route("/chat", methods=["GET"])
def chat_view():
    """
    Render a simple chat page that shows the current session-based conversation history.
    """
    history = _get_session_chat_history()
    return render_template("chat.html", history=history)

@api_bp.route("/chat", methods=["POST"])
def chat_post():
    """
    Session-based product chat.

    This chat does not answer generically.
    It answers based on the most recent deterministic system analysis
    stored in the current Flask session.

    Flow:
    1. read user message
    2. load engine_result + assessment context from session
    3. load chat history from session
    4. build a grounded prompt using the actual analysis result
    5. generate assistant response
    6. store both user and assistant messages in session
    7. optionally persist to DB if assessment_id exists in session
    """
    data = request.get_json(silent=True)

    if not data or not data.get("message"):
        return _error_response("Missing chat message", 400)

    user_message = data["message"].strip()
    if not user_message:
        return _error_response("Empty chat message", 400)

    engine_result = session.get("engine_result")
    assessment_context = session.get("assessment_context", {})
    last_summary = session.get("last_summary", "")

    if not engine_result:
        return _error_response(
            "No analysis context found. Please run an analysis first before using the chat.",
            400,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _error_response("OPENAI_API_KEY is not configured.", 500)

    history = _get_session_chat_history()
    history.append({"role": "user", "content": user_message})

    # Only keep recent history compact enough for prompt quality
    recent_history = history[-8:]

    history_block = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in recent_history
    )

    company_id = assessment_context.get("company_id", "Unknown company")
    industry = assessment_context.get("industry", "Unknown industry")

    scores = engine_result.get("dimension_scores", {})
    bottlenecks = engine_result.get("bottlenecks", [])
    overall_readiness = engine_result.get("overall_readiness", 0.0)
    transition_feasible = engine_result.get("transition_feasible", False)
    transition_risk = engine_result.get("transition_risk", "unknown")
    required_changes = engine_result.get("required_changes", {})
    required_capacities = engine_result.get("required_capacities", {})
    bottleneck_details = engine_result.get("bottleneck_details", {})
    maturity_descriptions = engine_result.get("maturity_descriptions", {})

    system_context = f"""
System analysis context for the current company:

Company: {company_id}
Industry: {industry}

Dimension scores:
{json.dumps(scores)}

Maturity descriptions:
{json.dumps(maturity_descriptions)}

Overall readiness:
{overall_readiness}

Transition feasible:
{transition_feasible}

Transition risk:
{transition_risk}

Bottlenecks:
{json.dumps(bottlenecks)}

Required changes:
{json.dumps(required_changes)}

Required capacities:
{json.dumps(required_capacities)}

Bottleneck details:
{json.dumps(bottleneck_details)}

Latest summary:
{last_summary}
""".strip()

    technology_hints = {
        "logistik": """
BRANCHEN-SPEZIFISCHE TECHNOLOGIE-EMPFEHLUNGEN (Logistik):
- Warehouse Management Systeme (WMS): SAP EWM, Manhattan Associates, Blue Yonder
- Transport Management Systeme (TMS): project44, Blue Yonder TMS, Transporeon
- Automatisierung: AMR (Autonomous Mobile Robots), Pick-by-Voice, RFID-Tracking
- Analyse: Real-Time-Tracking-Dashboards, Predictive-ETA-Engines
- Kommunikation: Fahrer-Apps mit einfacher UI (象大, TMW Systems)
- Schulung vor Automation: Mitarbeiter schulen BEVOR neue Systeme ausgerollt werden
""",
        "logistics": """
INDUSTRY-SPECIFIC TECHNOLOGY RECOMMENDATIONS (Logistics):
- Warehouse Management Systems (WMS): SAP EWM, Manhattan Associates, Blue Yonder
- Transport Management Systems (TMS): project44, Blue Yonder TMS, Transporeon
- Automation: AMR (Autonomous Mobile Robots), Pick-by-Voice, RFID tracking
- Analytics: Real-time tracking dashboards, predictive ETA engines
- Communication: Driver apps with simple UI
- Train BEFORE automation: Train employees BEFORE rolling out new systems
""",
        "produktion": """
BRANCHEN-SPEZIFISCHE TECHNOLOGIE-EMPFEHLUNGEN (Produktion):
- MES (Manufacturing Execution System): Siemens Opcenter, Rockwell FactoryTalk, SAP ME
- ERP-Integration: SAP S/4HANA Manufacturing, Oracle Cloud MES
- Qualitätssicherung: SPC-Software (Minitab, QI Macros), CAQ-Systeme
- Automatisierung: PLC-Programmierung, SCADA-Systeme, Cobots
- Wartung: Predictive Maintenance (Siemens MindSphere, PTC ThingWorx)
- WICHTIG: Erst Prozesse standardisieren (Lean), DANN automatisieren
""",
        "manufacturing": """
INDUSTRY-SPECIFIC TECHNOLOGY RECOMMENDATIONS (Manufacturing):
- MES (Manufacturing Execution System): Siemens Opcenter, Rockwell FactoryTalk, SAP ME
- ERP Integration: SAP S/4HANA Manufacturing, Oracle Cloud MES
- Quality Assurance: SPC software (Minitab, QI Macros), CAQ systems
- Automation: PLC programming, SCADA systems, Cobots
- Maintenance: Predictive Maintenance (Siemens MindSphere, PTC ThingWorx)
- IMPORTANT: Standardize processes (Lean) FIRST, THEN automate
""",
        "e-commerce": """
BRANCHEN-SPEZIFISCHE TECHNOLOGIE-EMPFEHLUNGEN (E-Commerce):
- Shop-Systeme: Shopify Plus, Magento (Adobe Commerce), Shopware
- ERP/PIM: SAP Commerce Cloud, Akeneo PIM, Clerk.io
- WMS: Bringg, Square Kiala, Packiyo
- Analytics: Google Analytics 4, Mixpanel, Heap
- Marketing: Klaviyo (E-Mail), Attentive (SMS), Attribrid
- Skalierbarkeit: Cloud-Infrastruktur (AWS, GCP) mit Auto-Scaling
""",
        "ecommerce": """
INDUSTRY-SPECIFIC TECHNOLOGY RECOMMENDATIONS (E-Commerce):
- Shop Systems: Shopify Plus, Magento (Adobe Commerce), Shopware
- ERP/PIM: SAP Commerce Cloud, Akeneo PIM
- WMS: Bringg, Packiyo
- Analytics: Google Analytics 4, Mixpanel, Heap
- Scalability: Cloud infrastructure (AWS, GCP) with auto-scaling
""",
        "pharma": """
BRANCHEN-SPEZIFISCHE TECHNOLOGIE-EMPFEHLUNGEN (Pharma):
- LIMS (Laboratory Information): STARLIMS, LabWare, LabVantage
- MES: Siemens Opcenter, Rockwell PharmaSuite, Werum PAS-X
- Serialisierung: TraceLink, Adents Proverb, SAP OEE für Track & Trace
- QMS (Quality Management): MasterControl, Veeva Vault QMS, Sparta Systems
- ERP: SAP S/4HANA (mit GxP-Modul), Oracle
- Compliance: Validierungsdokumentation, CSV (Computer System Validation)
- WICHTIG: Änderungen müssen validiert und dokumentiert sein (GMP)
""",
        "pharmaceutical": """
INDUSTRY-SPECIFIC TECHNOLOGY RECOMMENDATIONS (Pharma):
- LIMS: STARLIMS, LabWare, LabVantage
- MES: Siemens Opcenter, Rockwell PharmaSuite, Werum PAS-X
- Serialization: TraceLink, Adents Proverb
- QMS: MasterControl, Veeva Vault QMS, Sparta Systems
- ERP: SAP S/4HANA (with GxP module), Oracle
- Compliance: Validation documentation, CSV (Computer System Validation)
- IMPORTANT: Changes must be validated and documented (GMP)
""",
        "retail": """
BRANCHEN-SPEZIFISCHE TECHNOLOGIE-EMPFEHLUNGEN (Einzelhandel):
- POS-Systeme: Lightspeed POS, Square for Retail, Shopify POS
- Omnichannel: Shopify POS, Lightspeed Omnichannel, Salesfloor
- Warenwirtschaft: JTL-Wawi, SAP Retail, Oracle Xstore
- Kundenbindung: Yardo, Loyverse, Smile.io
- Analytics: Sensormatic (Footfall), Quanthub, Power BI Dashboards
- WICHTIG: Erst Mitarbeiter schulen, DANN neues Kassensystem ausrollen
- Change-Kommunikation: Filialleiter-Briefing vor Systemwechsel
""",
        "handel": """
BRANCHEN-SPEZIFISCHE TECHNOLOGIE-EMPFEHLUNGEN (Handel):
- POS-Systeme: Lightspeed POS, Square for Retail
- Omnichannel: Shopify POS, Lightspeed Omnichannel
- Inventory: JTL-Wawi, SAP Retail
- Customer loyalty: Yardo, Smile.io
- WICHTIG: Erst schulen, DANN Kassensystem ausrollen
""",
        "einzelhandel": """
BRANCHEN-SPEZIFISCHE TECHNOLOGIE-EMPFEHLUNGEN (Einzelhandel):
- POS-Systeme: Lightspeed POS, Square for Retail, Shopify POS
- Omnichannel: Shopify POS, Lightspeed Omnichannel
- Warenwirtschaft: JTL-Wawi, SAP Retail
- WICHTIG: Erst schulen, DANN System ausrollen
""",
        "saas": """
BRANCHEN-SPEZIFISCHE TECHNOLOGIE-EMPFEHLUNGEN (SaaS):
- APM (Application Performance): Datadog, New Relic, Dynatrace
- Error Tracking: Sentry, Bugsnag
- Feature Flags: LaunchDarkly, Unleash, Statsig
- CI/CD: GitHub Actions, GitLab CI, CircleCI
- Monitoring: PagerDuty, Grafana, Prometheus
- Onboarding: Appcues, WalkMe, Intercom Product Tours
- API-Management: Postman, Swagger, Kong
""",
        "tech": """
INDUSTRY-SPECIFIC TECHNOLOGY RECOMMENDATIONS (Tech):
- APM: Datadog, New Relic, Dynatrace
- Error Tracking: Sentry, Bugsnag
- Feature Flags: LaunchDarkly, Unleash
- CI/CD: GitHub Actions, GitLab CI
- Monitoring: PagerDuty, Grafana
- Onboarding: Appcues, WalkMe
- API Management: Postman, Swagger, Kong
""",
        "software": """
INDUSTRY-SPECIFIC TECHNOLOGY RECOMMENDATIONS (Software):
- APM: Datadog, New Relic, Dynatrace
- Feature Flags: LaunchDarkly, Unleash
- CI/CD: GitHub Actions, GitLab CI
- Onboarding: Appcues, WalkMe, Intercom
""",
    }

    industry_hint = ""
    industry_lower = industry.lower().strip() if industry else ""
    for key, hint in technology_hints.items():
        if key in industry_lower:
            industry_hint = hint
            break

    prompt = f"""
You are the built-in analyst of an Explainable Decision Engine.

You must answer ONLY based on the provided system analysis context.
Do not behave like a generic assistant.
Do not give generic consulting advice.
Do not tell the user to analyse things that are already known from the result.

Your role:
- explain the current result
- answer follow-up questions about bottlenecks, scores, priorities, risks, strengths, and next steps
- stay grounded in the actual company result
- be concrete and operational
- if the user asks "what is my strongest bottleneck", answer directly from the scores and bottlenecks
- if the user asks "what should I fix first", answer using the weakest dimensions and required capacities
- if the user asks about risks, explain them in relation to the current result
- if the user asks about technologies to introduce, use the industry-specific technology list below to give CONCRETE, NAMED recommendations
- if the user asks something unrelated to the analysis, answer briefly and redirect to the available system context

Style:
- concise
- direct
- no buzzwords
- no generic management fluff
- no numbered consulting frameworks unless explicitly helpful
- do not invent missing data

{system_context}
{f"""
Industry-specific technology recommendations:
{industry_hint}
""" if industry_hint else ""}
Conversation history:
{history_block}

Current user message:
{user_message}
""".strip()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "developer",
                    "content": (
                        "You are a product-embedded analyst for an explainable operational decision engine. "
                        "You must answer based on the supplied analysis result. "
                        "Never default to generic business advice when the system context already contains the answer. "
                        "Be specific, grounded, and concise. "
                        "IMPORTANT: Do NOT use markdown formatting. Write in plain text without **bold**, *italic*, or any other markdown syntax. "
                        "Use plain paragraphs and bullet points with hyphens (-)."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        assistant_message = response.output_text.strip()

        lang = _get_language()
        if lang == "de":
            assistant_message = _translate_to_german(assistant_message, api_key)

    except Exception as e:
        return _error_response("Chat generation failed", 500, str(e))

    history.append({"role": "assistant", "content": assistant_message})
    _save_session_chat_history(history)

    # Optional DB persistence if a session assessment id exists
    assessment_id = session.get("assessment_id")
    if assessment_id:
        try:
            _store_conversation_turn(
                assessment_id=assessment_id,
                role="user",
                content=user_message,
            )
            _store_conversation_turn(
                assessment_id=assessment_id,
                role="assistant",
                content=assistant_message,
            )
            db.session.commit()
        except Exception:
            db.session.rollback()

    return jsonify({
        "message": assistant_message,
        "history": history,
    }), 200

@api_bp.route("/chat/clear", methods=["POST"])
def clear_chat():
    """
    Clear the session-based chat history.
    """
    _clear_session_chat_history()
    return jsonify({"message": "Chat history cleared"}), 200

@api_bp.route("/temperature-demo", methods=["GET"])
def temperature_demo():
    """
    Render the temperature comparison demo page.

    This page lets the user compare two LLM outputs for the same assessment
    using different temperature values.
    """
    assessments = Assessment.query.all()
    lang = _get_language()

    if not assessments and session.get("engine_result"):
        assessments = ["session"]

    return render_template(
        "temperature_demo.html",
        assessments=assessments,
        lang=lang,
    )


@api_bp.route("/explanations/compare-temperature", methods=["POST"])
def compare_temperature():
    """
    Compare two explanation outputs for the same assessment using
    two different temperature settings.

    Expected JSON body:
    {
        "assessment_id": 1,
        "temperature_a": 0.2,
        "temperature_b": 0.8
    }
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    assessment_id = data.get("assessment_id")
    temperature_a = data.get("temperature_a")
    temperature_b = data.get("temperature_b")

    if assessment_id is None:
        return jsonify({"error": "assessment_id is required"}), 400

    if temperature_a is None or temperature_b is None:
        return jsonify({"error": "temperature_a and temperature_b are required"}), 400

    try:
        assessment_id = int(assessment_id)
        temperature_a = float(temperature_a)
        temperature_b = float(temperature_b)
    except (TypeError, ValueError):
        return jsonify({"error": "assessment_id must be int and temperatures must be numeric"}), 400

    if not (0 <= temperature_a <= 2) or not (0 <= temperature_b <= 2):
        return jsonify({"error": "temperature values must be between 0 and 2"}), 400

    engine_result = None
    assessment_context = {}
    
    if str(assessment_id) == "session":
        engine_result = session.get("engine_result")
        assessment_context = session.get("assessment_context", {})
        if not engine_result:
            return jsonify({"error": "No analysis in current session. Please run an analysis first."}), 400
    else:
        assessment = Assessment.query.get(assessment_id)
        if not assessment:
            return jsonify({"error": "Assessment not found"}), 404
        if not assessment.metrics:
            return jsonify({"error": "Assessment has no metrics"}), 404
        
        mapped_indicators = map_metrics_to_indicators(assessment.metrics)
        engine_result = run_deterministic_engine(mapped_indicators, assessment.target_level)
        assessment_context = {
            "company_id": assessment.domain or assessment.company_id or "Assessment",
            "industry": assessment.industry or "",
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY is not configured"}), 500

    try:
        retrieved_context = retrieve_context(engine_result)

        comparison_result = generate_temperature_comparison_openai(
            api_key=api_key,
            engine_result=engine_result,
            retrieved_context=retrieved_context,
            temperature_a=temperature_a,
            temperature_b=temperature_b,
        )

        return jsonify({
            "assessment_id": str(assessment_id),
            "assessment_name": assessment_context.get("company_id", "Session Analysis"),
            "industry": assessment_context.get("industry", ""),
            "temperature_a": temperature_a,
            "temperature_b": temperature_b,
            "engine_result": {
                "dimension_scores": engine_result.get("dimension_scores", {}),
                "bottlenecks": engine_result["bottlenecks"],
                "overall_readiness": engine_result.get("overall_readiness", 0),
                "transition_feasible": engine_result["transition_feasible"],
                "transition_risk": engine_result["transition_risk"],
            },
            "output_a": comparison_result["output_a"],
            "output_b": comparison_result["output_b"],
            "ai_summary": comparison_result.get("ai_summary", ""),
            "model_name": comparison_result["model_name"],
            "prompt_version": comparison_result["prompt_version"],
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Temperature comparison failed",
            "details": str(e),
        }), 500


@api_bp.route("/explanations/analyze-company", methods=["POST"])
def analyze_company():
    """
    Analyze a company using structured AI recommendations with temperature comparison.
    
    Expected JSON body:
    {
        "company_id": "Company Name",
        "industry": "Industry",
        "automation_rate": 70,
        "system_availability": 95,
        "error_rate": 5.0,
        "order_processing_time": 15,
        "process_standardization": "high",
        "role_clarity": "clear",
        "ownership_definition": "formal",
        "training_coverage": 60,
        "tool_adoption": 50,
        "change_communication": "structured",
        "temperature_a": 0.2,
        "temperature_b": 0.8
    }
    """
    import logging
    logger = logging.getLogger(__name__)
    
    data = request.get_json(silent=True)
    if not data:
        logger.error("Invalid or missing JSON body")
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    logger.info(f"Analyze company request: {data.get('company_id', 'Unknown')}")
    
    lang = _get_language()
    temperature_a = float(data.get("temperature_a", 0.2))
    temperature_b = float(data.get("temperature_b", 0.8))
    logger.info(f"Temperatures: A={temperature_a}, B={temperature_b}")

    try:
        metrics = [
            SimpleNamespace(name="automation_rate", value=float(data.get("automation_rate", 0))),
            SimpleNamespace(name="system_availability", value=float(data.get("system_availability", 0))),
            SimpleNamespace(name="error_rate", value=float(data.get("error_rate", 0))),
            SimpleNamespace(name="order_processing_time", value=float(data.get("order_processing_time", 0))),
            SimpleNamespace(name="process_standardization", value=data.get("process_standardization", "medium")),
            SimpleNamespace(name="role_clarity", value=data.get("role_clarity", "partial")),
            SimpleNamespace(name="ownership_definition", value=data.get("ownership_definition", "informal")),
            SimpleNamespace(name="training_coverage", value=float(data.get("training_coverage", 0))),
            SimpleNamespace(name="tool_adoption", value=float(data.get("tool_adoption", 0))),
            SimpleNamespace(name="change_communication", value=data.get("change_communication", "irregular")),
        ]

        mapped_indicators = map_metrics_to_indicators(metrics)
        logger.info(f"Mapped indicators: {mapped_indicators}")

        engine_result = run_deterministic_engine(mapped_indicators, target_level=2)
        logger.info(f"Engine result dimension_scores: {engine_result.get('dimension_scores')}")

        result = {
            "engine_result": engine_result,
            "temperature_a": temperature_a,
            "temperature_b": temperature_b,
            "priorities": [],
            "lever": None,
            "next_step": None,
            "summary_a": None,
            "summary_b": None,
        }

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                logger.info("Generating AI explanations with two temperatures...")
                retrieved_context = retrieve_context(engine_result)

                raw_output_a = generate_explanation_openai(
                    api_key=api_key,
                    engine_result=engine_result,
                    retrieved_context=retrieved_context,
                    industry=data.get("industry", ""),
                    temperature=temperature_a,
                )
                logger.info(f"Output A generated: summary length={len(raw_output_a.get('summary', ''))}")

                raw_output_b = generate_explanation_openai(
                    api_key=api_key,
                    engine_result=engine_result,
                    retrieved_context=retrieved_context,
                    industry=data.get("industry", ""),
                    temperature=temperature_b,
                )
                logger.info(f"Output B generated: summary length={len(raw_output_b.get('summary', ''))}")

                if lang == "de":
                    raw_output_a = _translate_structured_output(raw_output_a, api_key)
                    raw_output_b = _translate_structured_output(raw_output_b, api_key)

                result["summary_a"] = raw_output_a.get("summary", "")
                result["summary_b"] = raw_output_b.get("summary", "")

                result["priorities"] = raw_output_a.get("top_priorities", [])
                result["lever"] = raw_output_a.get("lever", {})
                result["next_step"] = raw_output_a.get("next_step", {})
                logger.info("AI explanations processed successfully")

            except Exception as e:
                logger.error(f"AI analysis failed: {e}", exc_info=True)
                result["summary_a"] = f"AI analysis failed: {str(e)}"
                result["summary_b"] = ""
        else:
            logger.warning("OPENAI_API_KEY not configured, returning engine result only")

        logger.info(f"Returning result with engine_result keys: {list(result['engine_result'].keys())}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Company analysis failed: {e}", exc_info=True)
        return jsonify({
            "error": "Company analysis failed",
            "details": str(e),
        }), 500