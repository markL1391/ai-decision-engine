"""
webapp.py – Flask App Factory

Verwendung:
    run.py ruft create_app() auf und startet den Server.
"""

from __future__ import annotations

import os
from datetime import datetime

from flask import Flask, render_template, request

from core.advisor import Advisor, AdvisorConfig
from core.prompt import MetricInput
from core.rag import SAMPLE_DOCUMENTS


def create_app() -> Flask:
    app = Flask(__name__)

    # ── Advisor einmalig initialisieren ───────────────────────────────────────
    config = AdvisorConfig(
        model="claude-opus-4-6",
        max_tokens=1024,
        rag_n_results=3,
        rag_enabled=True,
        target_level=2,
        chroma_path="./chroma_db",
    )
    advisor = Advisor(config=config)
    advisor.load_documents(SAMPLE_DOCUMENTS)

    # ── Routen ────────────────────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/demo", methods=["POST"])
    def demo():
        # Formular-Werte einlesen
        metrics = _parse_form(request.form)

        company_id = request.form.get("company_id", "Unbekanntes Unternehmen")
        industry   = request.form.get("industry", None)
        assessed_at = datetime.today().strftime("%d.%m.%Y")

        # Advisor-Durchlauf
        result = advisor.run(
            metrics=metrics,
            company_id=company_id,
            industry=industry,
            assessed_at=assessed_at,
        )

        # Template-Kontext aufbauen
        ctx = _build_template_context(result)
        return render_template("index.html", **ctx)

    return app


# ── Formular-Parser ───────────────────────────────────────────────────────────

def _parse_form(form) -> list[MetricInput]:
    """Liest alle KPI-Felder aus dem POST-Formular."""
    return [
        MetricInput(name="automation_rate",         value=float(form["automation_rate"])),
        MetricInput(name="system_availability",     value=float(form["system_availability"])),
        MetricInput(name="error_rate",              value=float(form["error_rate"])),
        MetricInput(name="order_processing_time",   value=float(form["order_processing_time"])),
        MetricInput(name="process_standardization", value=form["process_standardization"]),
        MetricInput(name="role_clarity",            value=form["role_clarity"]),
        MetricInput(name="ownership_definition",    value=form["ownership_definition"]),
        MetricInput(name="training_coverage",       value=float(form["training_coverage"])),
        MetricInput(name="tool_adoption",           value=float(form["tool_adoption"])),
        MetricInput(name="change_communication",    value=form["change_communication"]),
    ]


# ── Template-Kontext ──────────────────────────────────────────────────────────

def _build_template_context(advisor_result) -> dict:
    """
    Baut den Dict, den index.html erwartet.
    Nutzt structured (AdvisorOutput) wenn verfügbar, sonst Fallback auf Rohtext.
    """
    engine   = advisor_result.engine_output
    structured = advisor_result.structured

    # Basis-Ergebnis (deterministisch, immer vorhanden)
    ctx = {
        "result": engine,
        "assessment": advisor_result.assessment,
    }

    if structured:
        ctx["summary"]     = structured.summary
        ctx["priorities"]  = [p.model_dump() for p in structured.top_priorities]
        ctx["lever"]       = structured.lever.model_dump()
        ctx["risk"]        = structured.risk.model_dump()
        ctx["next_step"]   = structured.next_step.model_dump()
        ctx["references"]  = structured.rag_references
    else:
        # Fallback: Rohtext aus LLM
        ctx["summary"]    = advisor_result.recommendation
        ctx["priorities"] = []
        ctx["lever"]      = None
        ctx["risk"]       = None
        ctx["next_step"]  = None
        ctx["references"] = []

    return ctx
