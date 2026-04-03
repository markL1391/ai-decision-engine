"""
advisor.py – Hauptorchestrator des Maturity Assessment Advisors

Verbindet:
  mapping.py  → KPI-Mapping
  engine.py   → Deterministische Bewertung
  rag.py      → Kontextabruf
  prompt.py   → Prompt-Aufbau
  LLM-API     → Empfehlung generieren

Installation:
    pip install anthropic chromadb sentence-transformers pydantic

Verwendung:
    from advisor import Advisor, AdvisorConfig
    advisor = Advisor()
    result  = advisor.run(metrics, company_id="Muster GmbH", industry="Logistik")
     print(result.recommendation)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from core.engine import run_deterministic_engine
from core.mapping import map_metrics_to_indicators
from core.output import AdvisorOutput, OutputParseError, inject_output_instruction, parse_llm_output
from core.prompt import SYSTEM_PROMPT, AssessmentInput, MetricInput, build_prompt_messages
from core.rag import RAGStore, build_rag_query


# ── Konfiguration ─────────────────────────────────────────────────────────────

@dataclass
class AdvisorConfig:
    """Alle einstellbaren Parameter an einem Ort."""
    model:          str   = "claude-opus-4-6"
    max_tokens:     int   = 1024
    rag_n_results:  int   = 3          # wie viele RAG-Chunks pro Call
    rag_enabled:    bool  = True
    target_level:   int   = 2          # Standard-Ziel-Reifegrad
    chroma_path:    str   = "./chroma_db"


# ── Ergebnis-Datenklasse ──────────────────────────────────────────────────────

@dataclass
class AdvisorResult:
    """Rückgabe eines vollständigen Advisor-Durchlaufs."""
    recommendation:  str                        # LLM-generierter Rohtext
    structured:      Optional[AdvisorOutput]    # Validierter, typisierter Output
    engine_output:   Dict[str, Any]             # Rohdaten der Engine
    rag_context:     str                        # verwendeter RAG-Kontext
    prompt_messages: List[Dict[str, str]]       # gesendete Messages (Debugging)
    assessment:      AssessmentInput            # Input-Snapshot

    def summary(self) -> str:
        """Kurzüberblick Engine-Scores für Logs oder CLI."""
        scores  = self.engine_output.get("dimension_scores", {})
        risk    = self.engine_output.get("transition_risk", "?")
        overall = self.engine_output.get("overall_readiness", 0.0)
        score_str = "  ".join(f"{d}: {v:.1f}" for d, v in scores.items())
        return (
            f"Unternehmen : {self.assessment.company_id}\n"
            f"Scores      : {score_str}\n"
            f"Bereitschaft: {overall:.1f} / 3\n"
            f"Risiko      : {risk}\n"
        )

    def recommendation_text(self) -> str:
        """Gibt strukturierten Empfehlungstext zurück (bevorzugt strukturiert)."""
        if self.structured:
            return self.structured.summary_text()
        return self.recommendation


# ── Advisor ───────────────────────────────────────────────────────────────────

class Advisor:
    """
    Hauptklasse – orchestriert den gesamten Bewertungs- und Empfehlungsprozess.

    Typischer Ablauf:
        advisor = Advisor()
        result  = advisor.run(metrics, company_id="Muster GmbH")
        print(result.recommendation)
    """

    def __init__(self, config: Optional[AdvisorConfig] = None) -> None:
        self.config  = config or AdvisorConfig()
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._rag = RAGStore(persist_path=self.config.chroma_path) if self.config.rag_enabled else None

    # ── Haupt-Methode ─────────────────────────────────────────────────────────

    def run(
        self,
        metrics: List[MetricInput],
        company_id: str,
        industry: Optional[str] = None,
        target_level: Optional[int] = None,
        assessed_at: Optional[str] = None,
        rag_where: Optional[dict] = None,
    ) -> AdvisorResult:
        """
        Führt einen vollständigen Bewertungs-Durchlauf durch.

        Args:
            metrics:      Liste von MetricInput-Objekten (aus mapping.py)
            company_id:   Unternehmensname oder ID
            industry:     Branche (optional, verbessert RAG-Relevanz)
            target_level: Ziel-Reifegrad 0–3 (default aus Config)
            assessed_at:  Datum als String, z.B. "03.04.2026"
            rag_where:    Optionaler Metadaten-Filter für RAG-Abfrage

        Returns:
            AdvisorResult mit Empfehlung, Engine-Output und Debug-Daten
        """
        level = target_level or self.config.target_level

        # 1 – Assessment-Input aufbauen
        assessment = AssessmentInput(
            company_id=company_id,
            industry=industry,
            target_level=level,
            assessed_at=assessed_at,
            metrics=metrics,
        )

        # 2 – KPI-Mapping + Engine
        indicators    = map_metrics_to_indicators(metrics)
        engine_output = run_deterministic_engine(indicators, target_level=level)

        # 3 – RAG-Kontext abrufen
        rag_context = ""
        if self._rag:
            query       = build_rag_query(engine_output, industry=industry)
            rag_context = self._rag.retrieve(
                query,
                n_results=self.config.rag_n_results,
                where=rag_where,
            )

        # 4 – Prompt aufbauen
        messages = build_prompt_messages(assessment, engine_output, rag_context)

        # 5 – System-Prompt um Output-Anweisung ergänzen
        messages[0]["content"] = inject_output_instruction(messages[0]["content"])

        # 6 – LLM-Call
        recommendation = self._call_llm(messages)

        # 7 – Strukturierten Output parsen
        structured: Optional[AdvisorOutput] = None
        try:
            structured = parse_llm_output(recommendation)
        except OutputParseError as e:
            print(f"[Advisor] Parse-Warnung: {e} – Rohtext wird verwendet.")

        return AdvisorResult(
            recommendation=recommendation,
            structured=structured,
            engine_output=engine_output,
            rag_context=rag_context,
            prompt_messages=messages,
            assessment=assessment,
        )

    # ── LLM-Call ──────────────────────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Sendet die Messages an die Anthropic API.
        System-Message wird separat übergeben (Anthropic-Format).
        """
        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_messages = [m for m in messages if m["role"] != "system"]

        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_msg,
            messages=user_messages,
        )
        return response.content[0].text

    # ── RAG-Dokumente laden ───────────────────────────────────────────────────

    def load_documents(self, documents: list) -> None:
        """Fügt Dokumente zur RAG-Datenbank hinzu (einmalig beim Setup)."""
        if self._rag:
            self._rag.add_documents(documents)
        else:
            print("[Advisor] RAG ist deaktiviert – Dokumente werden nicht geladen.")


# ── CLI / Quick-Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag import SAMPLE_DOCUMENTS

    # Advisor initialisieren
    advisor = Advisor(config=AdvisorConfig(
        model="claude-opus-4-6",
        max_tokens=1024,
        rag_n_results=3,
    ))

    # Beispiel-Dokumente einmalig laden
    advisor.load_documents(SAMPLE_DOCUMENTS)

    # Beispiel-Metriken (Logistik-Unternehmen mit Schwächen in P und A)
    metrics = [
        MetricInput(name="automation_rate",        value=55.0),
        MetricInput(name="system_availability",    value=96.0),
        MetricInput(name="error_rate",             value=8.0),
        MetricInput(name="order_processing_time",  value=22.0),
        MetricInput(name="process_standardization",value="medium"),
        MetricInput(name="role_clarity",           value="partial"),
        MetricInput(name="ownership_definition",   value="informal"),
        MetricInput(name="training_coverage",      value=45.0),
        MetricInput(name="tool_adoption",          value=38.0),
        MetricInput(name="change_communication",   value="irregular"),
    ]

    print("Advisor wird gestartet...\n")
    result = advisor.run(
        metrics=metrics,
        company_id="Muster GmbH",
        industry="Logistik",
        target_level=2,
        assessed_at="03.04.2026",
    )

    print(result.summary())
    print(result.recommendation_text())
