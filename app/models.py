"""
models.py

SQLAlchemy data models for the Explainable AI Maturity Assessment System.

This module defines the persistence layer for:
- assessment metadata
- raw metric inputs
- mapped indicator scores
- deterministic engine results
- generated AI explanations
- retained conversation history

These tables support the project requirements for:
- SQLite persistence
- inserting and reading entries
- updating the database after text generation
- retaining conversation history
"""

from datetime import datetime

from app import db


class Assessment(db.Model):
    """
    Main parent entity for one maturity assessment run.

    Stores the high-level metadata of an assessment and connects all
    dependent records such as raw metrics, indicator scores, results,
    explanations, and conversation history.
    """
    __tablename__ = "assessments"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    domain = db.Column(db.String(100), nullable=False, default="generic")
    notes = db.Column(db.Text, nullable=True)
    target_level = db.Column(db.Integer, nullable=False, default=2)

    # One assessment can contain many raw metric inputs.
    metrics = db.relationship(
        "Metric",
        backref="assessment",
        lazy=True,
        cascade="all, delete-orphan",
    )

    # One assessment can contain many mapped indicator scores.
    indicator_scores = db.relationship(
        "IndicatorScore",
        backref="assessment",
        lazy=True,
        cascade="all, delete-orphan",
    )

    # One assessment has exactly one deterministic result record.
    result = db.relationship(
        "Result",
        backref="assessment",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # One assessment can have one persisted explanation record.
    explanation = db.relationship(
        "Explanation",
        backref="assessment",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # One assessment can have many conversation turns.
    conversation_turns = db.relationship(
        "ConversationTurn",
        backref="assessment",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="ConversationTurn.created_at",
    )

    def __repr__(self) -> str:
        """Readable debug representation."""
        return (
            f"<Assessment id={self.id} domain={self.domain!r} "
            f"target_level={self.target_level}>"
        )


class Metric(db.Model):
    """
    Raw input metric provided by the user or frontend.

    Example:
    - automation_rate
    - system_availability
    - error_rate
    """
    __tablename__ = "metrics"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(
        db.Integer,
        db.ForeignKey("assessments.id"),
        nullable=False,
    )
    name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.String, nullable=False)
    unit = db.Column(db.String(50), nullable=True)

    def __repr__(self) -> str:
        """Readable debug representation."""
        return f"<Metric id={self.id} name={self.name!r} value={self.value!r}>"


class IndicatorScore(db.Model):
    """
    Normalized score derived from raw metrics.

    Each indicator belongs to one maturity dimension:
    - T = Technology
    - P = Process
    - R = Responsibility
    - A = Adoption

    Values are standardized to a 0-3 maturity scale.
    """
    __tablename__ = "indicator_scores"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(
        db.Integer,
        db.ForeignKey("assessments.id"),
        nullable=False,
    )
    dimension = db.Column(db.String(1), nullable=False)   # R, P, T, A
    indicator = db.Column(db.String(10), nullable=False)  # e.g. T1, P2
    value = db.Column(db.Integer, nullable=False)         # 0-3

    def __repr__(self) -> str:
        """Readable debug representation."""
        return (
            f"<IndicatorScore id={self.id} dimension={self.dimension!r} "
            f"indicator={self.indicator!r} value={self.value}>"
        )


class Result(db.Model):
    """
    Deterministic engine result for one assessment.

    Stores aggregated maturity scores and the key outcome fields generated
    by the deterministic engine, such as bottlenecks and transition risk.
    """
    __tablename__ = "results"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(
        db.Integer,
        db.ForeignKey("assessments.id"),
        nullable=False,
        unique=True,
    )

    r_score = db.Column(db.Float, nullable=False)
    p_score = db.Column(db.Float, nullable=False)
    t_score = db.Column(db.Float, nullable=False)
    a_score = db.Column(db.Float, nullable=False)
    overall_readiness = db.Column(db.Float, nullable=False)

    bottlenecks_json = db.Column(db.Text, nullable=False)
    transition_feasible = db.Column(db.Boolean, nullable=False)
    transition_risk = db.Column(db.String(20), nullable=False)
    required_changes_json = db.Column(db.Text, nullable=False)

    def __repr__(self) -> str:
        """Readable debug representation."""
        return (
            f"<Result id={self.id} assessment_id={self.assessment_id} "
            f"overall_readiness={self.overall_readiness}>"
        )


class Explanation(db.Model):
    """
    Persisted AI-generated explanation for one assessment.

    This table is updated by the text generation endpoint and stores the
    structured explanation parts returned by the LLM.
    """
    __tablename__ = "explanations"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(
        db.Integer,
        db.ForeignKey("assessments.id"),
        nullable=False,
        unique=True,
    )

    why_limit_json = db.Column(db.Text, nullable=False)
    blocks_transition_json = db.Column(db.Text, nullable=False)
    references_json = db.Column(db.Text, nullable=True)
    model_name = db.Column(db.String(100), nullable=True)
    prompt_version = db.Column(db.String(50), nullable=True)

    def __repr__(self) -> str:
        """Readable debug representation."""
        return (
            f"<Explanation id={self.id} assessment_id={self.assessment_id} "
            f"model={self.model_name!r}>"
        )


class ConversationTurn(db.Model):
    """
    Stores retained conversation history for one assessment.

    This supports the project requirement for conversation memory by saving:
    - prompts
    - prior assistant outputs
    - optional system context

    These entries can later be reloaded and injected into future prompts.
    """
    __tablename__ = "conversation_turns"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(
        db.Integer,
        db.ForeignKey("assessments.id"),
        nullable=False,
    )
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    def __repr__(self) -> str:
        """Readable debug representation."""
        return (
            f"<ConversationTurn id={self.id} assessment_id={self.assessment_id} "
            f"role={self.role!r}>"
        )