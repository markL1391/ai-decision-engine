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

from flask import Blueprint, jsonify, render_template, request
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
)
from core.engine import run_deterministic_engine
from core.mapping import map_metrics_to_indicators
from core.rag import retrieve_context
from core.output import OutputParseError, parse_llm_output

api_bp = Blueprint("api", __name__)

DIMENSION_DISPLAY = {
    "T": "Technology",
    "P": "Process",
    "R": "Responsibility",
    "A": "Adoption",
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


def _build_comparison_payload(result_a: Dict[str, Any], result_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a comparison dictionary from two deterministic assessment results.

    Args:
        result_a: Deterministic result of assessment A.
        result_b: Deterministic result of assessment B.

    Returns:
        Dictionary containing score deltas and feasibility comparison.
    """
    return {
        "r_delta": round(result_b["dimension_scores"].get("R", 0) - result_a["dimension_scores"].get("R", 0), 2),
        "p_delta": round(result_b["dimension_scores"].get("P", 0) - result_a["dimension_scores"].get("P", 0), 2),
        "t_delta": round(result_b["dimension_scores"].get("T", 0) - result_a["dimension_scores"].get("T", 0), 2),
        "a_delta": round(result_b["dimension_scores"].get("A", 0) - result_a["dimension_scores"].get("A", 0), 2),
        "overall_readiness_delta": round(result_b["overall_readiness"] - result_a["overall_readiness"], 2),
        "bottleneck_a": result_a["bottlenecks"],
        "bottleneck_b": result_b["bottlenecks"],
        "transition_feasible_a": result_a["transition_feasible"],
        "transition_feasible_b": result_b["transition_feasible"],
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


# =============================================================================
# Frontend / demo routes
# =============================================================================

@api_bp.route("/", methods=["GET"])
def home():
    """
    Render the main HTML frontend.

    This route exists mainly for demo and presentation purposes.
    """
    return render_template("index.html")


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
    metrics = _build_demo_metrics(request.form)
    mapped_indicators = map_metrics_to_indicators(metrics)
    result = run_deterministic_engine(mapped_indicators, target_level=2)

    assessment = SimpleNamespace(
        company_id=request.form.get("company_id", "Demo Company"),
        industry=request.form.get("industry"),
        target_level=2,
    )

    summary = _build_demo_summary(result)
    priorities = []
    lever = None
    risk = None
    next_step = None
    references = []

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            retrieved_context = retrieve_context(result)

            raw_output = generate_explanation_openai(
                api_key=api_key,
                engine_result=result,
                retrieved_context=retrieved_context,
            )

            structured = parse_llm_output(json.dumps({
                "summary": raw_output["summary"],
                "top_priorities": raw_output["top_priorities"],
                "lever": raw_output["lever"],
                "risk": raw_output["risk"],
                "next_step": raw_output["next_step"],
                "rag_references": raw_output["rag_references"],
            }))

            summary = structured.summary

            priorities = []
            for p in structured.top_priorities:
                item = p.model_dump()
                item["dimension_label"] = DIMENSION_DISPLAY.get(item["dimension"], item["dimension"])
                priorities.append(item)

            lever = structured.lever.model_dump()
            lever["dimension_label"] = DIMENSION_DISPLAY.get(lever["dimension"], lever["dimension"])

            risk = structured.risk.model_dump()
            next_step = structured.next_step.model_dump()

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
    )

@api_bp.route("/compare-demo", methods=["GET"])
def compare_demo():
    """
    Render a predefined comparison demo in the frontend.

    This route is useful for showcasing use-case-specific comparative analysis
    without requiring manual data entry first.
    """
    case_a = [
        SimpleNamespace(name="automation_rate", value=28.0),
        SimpleNamespace(name="system_availability", value=91.0),
        SimpleNamespace(name="error_rate", value=11.5),
        SimpleNamespace(name="order_processing_time", value=24.0),
        SimpleNamespace(name="process_standardization", value="low"),
        SimpleNamespace(name="role_clarity", value="partial"),
        SimpleNamespace(name="ownership_definition", value="informal"),
        SimpleNamespace(name="training_coverage", value=40.0),
        SimpleNamespace(name="tool_adoption", value=35.0),
        SimpleNamespace(name="change_communication", value="irregular"),
    ]

    case_b = [
        SimpleNamespace(name="automation_rate", value=55.0),
        SimpleNamespace(name="system_availability", value=96.0),
        SimpleNamespace(name="error_rate", value=5.2),
        SimpleNamespace(name="order_processing_time", value=15.0),
        SimpleNamespace(name="process_standardization", value="medium"),
        SimpleNamespace(name="role_clarity", value="clear"),
        SimpleNamespace(name="ownership_definition", value="formal"),
        SimpleNamespace(name="training_coverage", value=68.0),
        SimpleNamespace(name="tool_adoption", value=62.0),
        SimpleNamespace(name="change_communication", value="structured"),
    ]

    mapped_a = map_metrics_to_indicators(case_a)
    mapped_b = map_metrics_to_indicators(case_b)

    result_a = run_deterministic_engine(mapped_a, target_level=2)
    result_b = run_deterministic_engine(mapped_b, target_level=2)

    comparison = _build_comparison_payload(result_a, result_b)

    compare_explanation = None
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            compare_explanation = generate_compare_explanation_openai(
                api_key=api_key,
                comparison=comparison,
            )
        except Exception as e:
            compare_explanation = {
                "summary": "The comparison explanation could not be generated.",
                "main_improvements": [str(e)],
                "transition_impact": [],
            }

    return render_template(
        "index.html",
        compare_mode=True,
        result_a=result_a,
        result_b=result_b,
        comparison=comparison,
        compare_explanation=compare_explanation,
    )


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
            item["dimension_label"] = DIMENSION_DISPLAY.get(item["dimension"], item["dimension"])
            priority_items.append(item)

        lever_item = structured.lever.model_dump()
        lever_item["dimension_label"] = DIMENSION_DISPLAY.get(lever_item["dimension"], lever_item["dimension"])

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