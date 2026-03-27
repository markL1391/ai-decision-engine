from datetime import datetime
from email.policy import default

from sqlalchemy import nullsfirst

from app import db

class Assessment(db.Model):
    __tablename__ = "assessments"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    domain = db.Column(db.String(100), nullable=False, default="generic")
    notes = db.Column(db.Text, nullable=True)
    target_level = db.Column(db.Integer, nullable=False, default=2)

    metrics = db.relationship("Metric", backref="assessment", lazy=True, cascade="all, delete-orphan")
    indicator_scores = db.relationship("IndicatorScore", backref="assessment", uselist=False, cascade="all, delete-orphan")
    result = db.relationship("Result", backref="assessment", uselist=False, cascade="all, delete-orphan")

class Metric(db.Model):
    __tablename__ = "metrics"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessments.id"), nullable=False)
    name= db.Column(db.String(100), nullable=False)
    value = db.Column(db.String, nullable=False)
    unit = db.Column(db.String(50), nullable=True)

class IndicatorScore(db.Model):
    __tablename__ = "indicator_scores"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessments.id"), nullable=False)
    dimension =db.Column(db.String(1), nullable=False)      # R, P, T, A
    indicator = db.Column(db.String(10), nullable=False)    # e.g. T1, P2, etc.
    value = db.Column(db.Integer, nullable=False)   #0-3

class Result(db.Model):
    __tablename__ = "results"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessments.id"), nullable=False, unique=True)

    r_score = db.Column(db.Float, nullable=False)
    p_score = db.Column(db.Float, nullable=False)
    t_score = db.Column(db.Float, nullable=False)
    a_score = db.Column(db.Float, nullable=False)
    overall_readiness = db.Column(db.Float, nullable=False)

    bottlenecks_json = db.Column(db.Text, nullable=False)
    transition_feasible = db.Column(db.Boolean, nullable=False)
    transition_risk = db.Column(db.String(20), nullable=False)
    required_changes_json = db.Column(db.Text, nullable=False)

class Explanation(db.Model):
    __tablename__ = "explanations"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessments.id"), nullable=False, unique=True)

    why_limit_json = db.Column(db.Text, nullable=False)
    blocks_transition_json = db.Column(db.Text, nullable=False)
    references_json = db.Column(db.String(100), nullable=True)
    model_name = db.Column(db.String(100), nullable=True)
    prompt_version = db.Column(db.String(50), nullable=True)

class ConversationTurn(db.Model):
    __tablename__ = "conversation_turns"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessments.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
