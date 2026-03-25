import json
from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app import db
from app.models import Assessment, Metric, IndicatorScore, Result
from app.schemas import AssessmentAnalyzeRequest
from app.mapping import map_metrics_to_indicators

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

    for item in mapped_indicators:
        db.session.add(
            IndicatorScore(
                assessment_id=assessment.id,
                dimension=item["dimensions"],
                dummy_indicators=item["indicator"],
                value=item["value"],
            )
        )

    # 4. Dummy result for skeleton version
    result = Result(
        assessment_id=assessment.id,
        r_score=2.0,
        p_score=2.0,
        t_score=2.0,
        a_score=1.0,
        overall_readiness=1.0,
        bottlenecks_json=json.dumps(["A"]),
        transition_feasible=False,
        transition_risk="high",
        required_changes_json=json.dumps({"A": ["A1", "A2"]}),
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