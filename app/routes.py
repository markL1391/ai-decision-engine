import json
import os

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app import db
from app.models import Assessment, Metric, IndicatorScore, Result, Explanation
from app.schemas import AssessmentAnalyzeRequest, ExplanationGenerateRequest, CompareAssessmentsRequest
from app.mapping import map_metrics_to_indicators
from app.engine import run_deterministic_engine
from app.rag import retrieve_context
from app.llm import build_prompt, generate_explanation_openai, build_compare_prompt, generate_compare_explanation_openai

api_bp = Blueprint("api", __name__)

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@api_bp.route("/assessments/analyze", methods=["POST"])
def analyze_assessment():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        payload = AssessmentAnalyzeRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 422

    # 1. Create assessment
    assessment = Assessment(
        domain=payload.assessment.domain,
        notes=payload.assessment.notes,
        target_level=payload.assessment.target_level,
    )
    db.session.add(assessment)
    db.session.flush()      # Get assessment.id before commit

    # 2. Save raw metrics
    for metric in payload.metrics:
        db_metric = Metric(
            assessment_id=assessment.id,
            name=metric.name,
            value=metric.value,
            unit=metric.unit,
        )
        db.session.add(db_metric)

    # 3. Indicator mapping
    mapped_indicators = map_metrics_to_indicators(payload.metrics)
    engine_result = run_deterministic_engine(
        mapped_indicators,
        payload.assessment.target_level
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

    # 4. Result
    result = Result(
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
    db.session.add(result)

    db.session.commit()

    return jsonify({
        "message": "Assessment created and analyzed successfully",
        "assessment_id": assessment.id,
    }), 201

@api_bp.route("/assessments/<int:assessment_id>", methods=["GET"])
def get_assessment(assessment_id: int):
    assessment = Assessment.query.get(assessment_id)

    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404

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
    return jsonify({
        "id": assessment.id,
        "created_at": assessment.created_at.isoformat(),
        "domain": assessment.domain,
        "notes": assessment.notes,
        "target_level": assessment.target_level,
        "metrics": metrics,
        "indicator_scores": indicators,
        "results": result,
    }), 200

@api_bp.route("/explanations/generate", methods=["POST"])
def generate_explanation():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        payload = ExplanationGenerateRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 422

    assessment = Assessment.query.get(payload.assessment_id)
    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404

    if not assessment.result:
        return jsonify({"error": "No deterministic result found for assessment"}), 404

    engine_result = {
        "dimension_scores": {
            "R": assessment.result.r_score,
            "P": assessment.result.p_score,
            "T": assessment.result.t_score,
            "A": assessment.result.a_score,
        },
        "overall_readiness": assessment.result.overall_readiness,
        "bottlenecks": json.loads(assessment.result.bottlenecks_json),
        "transition_feasible": assessment.result.transition_feasible,
        "transition_risk": assessment.result.transition_risk,
        "required_changes": json.loads(assessment.result.required_changes_json),
    }

    retrieved_context = retrieve_context(engine_result)
    prompt = build_prompt(engine_result, retrieved_context)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY is not configured."}), 500

    try:
        explanation_data = generate_explanation_openai(
            api_key=api_key,
            engine_result=engine_result,
            retrieved_context=retrieved_context,
        )

    except Exception as e:
        return jsonify({"error": "OpenAI generation failed", "details": str(e)}), 500

    existing = Explanation.query.filter_by(assessment_id=assessment.id).first()

    if existing:

        existing.why_limit_json = json.dumps(explanation_data["why_limit"])
        existing.blocks_transition_json = json.dumps(explanation_data["blocks_transition"])
        existing.references_json = json.dumps(explanation_data["references"])
        existing.model_name = explanation_data["model_name"]
        existing.prompt_version = explanation_data["prompt_version"]
    else:
        explanation = Explanation(
            assessment_id=assessment.id,
            why_limit_json=json.dumps(explanation_data["why_limit"]),
            blocks_transition_json=json.dumps(explanation_data["blocks_transition"]),
            references_json=json.dumps(explanation_data["references"]),
            model_name=explanation_data["model_name"],
            prompt_version=explanation_data["prompt_version"],
        )
        db.session.add(explanation)

    db.session.commit()

    return jsonify({
        "assessment_id": assessment.id,
        "prompt_preview": prompt,
        "explanation": {
            "why_limit": explanation_data["why_limit"],
            "blocks_transition": explanation_data["blocks_transition"],
            "references": explanation_data["references"],
        }
    }),200

@api_bp.route("/assessments/compare", methods=["POST"])
def compare_assessments():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        payload = CompareAssessmentsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 422

    a = Assessment.query.get(payload.assessment_a_id)
    b = Assessment.query.get(payload.assessment_b_id)

    if not a or not b:
        return jsonify({"error": "One or both assessments not found"}), 404

    if not a.result or not b.result:
        return jsonify({"error": "One or both assessments have no results."}), 404

    comparison = {
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

    return jsonify({
        "assessment_a_id": a.id,
        "assessment_b_id": b.id,
        "comparison": comparison
    }), 200

@api_bp.route("/assessments/compare/explain", methods=["POST"])
def compare_and_explain():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        payload = CompareAssessmentsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 422

    a = Assessment.query.get(payload.assessment_a_id)
    b = Assessment.query.get(payload.assessment_b_id)

    if not a or not b:
        return jsonify({"error": "One or both assessments not found"}), 404

    if not a.result or not b.result:
        return jsonify({"error": "One or both assessments have no results"}), 400

    comparison = {
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

    # LLM explanation
    prompt_preview = build_compare_prompt(comparison)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        explanation = generate_compare_explanation_openai(api_key, comparison)

    return jsonify({
        "assessment_a_id": a.id,
        "assessment_b_id": b.id,
        "comparison": comparison,
        "prompt_preview": prompt_preview,
        "explanation": explanation,
    }), 200


