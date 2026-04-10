"""
llm.py

LLM integration layer for the Explainable AI Maturity Assessment System.

This module is responsible for:
- building prompts for structured advisory output
- building prompts for comparative analysis explanations
- calling the OpenAI Responses API
- enforcing structured JSON output via JSON Schema
- returning parsed Python dictionaries to the route layer

The design principle is:
- deterministic logic decides
- the LLM explains and prioritizes in business language

This module supports:
- structured output
- prompt engineering
- comparative analysis
- frontend-ready advisory content
"""

import json
from typing import Any, Dict, List

from openai import OpenAI


# =============================================================================
# Constants
# =============================================================================

ADVISOR_MODEL_NAME = "gpt-4o-mini"
ADVISOR_PROMPT_VERSION = "v7_advisory_output_schema_grounded"

COMPARE_MODEL_NAME = "gpt-4o-mini"
COMPARE_PROMPT_VERSION = "v5_compare_operational_schema"


# =============================================================================
# Helper functions
# =============================================================================

def _build_context_block(retrieved_context: List[str]) -> str:
    """
    Convert retrieved context lines into a bullet-formatted block.

    Args:
        retrieved_context: Retrieved context lines from the retrieval layer
            and optional conversation history.

    Returns:
        Multiline bullet list string.
    """
    if not retrieved_context:
        return "- No additional context available."

    return "\n".join(f"- {item}" for item in retrieved_context)


def _safe_json_loads(raw_text: str) -> Dict[str, Any] | None:
    """
    Safely parse JSON returned by the model.

    Args:
        raw_text: Raw model output text.

    Returns:
        Parsed dictionary if valid JSON, otherwise None.
    """
    try:
        return json.loads(raw_text)
    except Exception:
        return None


# =============================================================================
# Prompt builders
# =============================================================================

def build_prompt(engine_result: Dict[str, Any], retrieved_context: List[str], industry: str = "") -> str:
    """
    Build the user prompt for structured advisory output.

    The model is asked to turn deterministic assessment data into a
    concrete, operational advisory response that can be rendered in the UI.

    Args:
        engine_result: Deterministic engine output.
        retrieved_context: Context lines from the retrieval layer and/or
            conversation history.
        industry: Industry context for branch-specific recommendations.

    Returns:
        Prompt string for the OpenAI Responses API.
    """
    scores = engine_result["dimension_scores"]
    overall = engine_result["overall_readiness"]
    bottlenecks = engine_result["bottlenecks"]
    feasible = engine_result["transition_feasible"]
    risk = engine_result["transition_risk"]
    required_changes = engine_result["required_changes"]
    required_capacities = engine_result["required_capacities"]
    bottleneck_details = engine_result["bottleneck_details"]
    maturity_descriptions = engine_result.get("maturity_descriptions", {})
    strengths = engine_result.get("strengths", [])
    weaknesses = engine_result.get("weaknesses", [])
    cross_dimension_insights = engine_result.get("cross_dimension_insights", [])
    executive_summary = engine_result.get("executive_summary", "")

    context_block = _build_context_block(retrieved_context)

    industry_hints = {
        "logistik": "Branchenkontext Logistik: Neue Technologien (WMS, TMS) scheitern oft an Schulungsmängeln. Erst trainieren, DANN ausrollen.",
        "logistics": "Industry context Logistics: New technologies (WMS, TMS) often fail due to training gaps. Train first, THEN roll out.",
        "produktion": "Branchenkontext Produktion: Maschinenausfall hat direkte Kostenfolgen. Erst standardisieren, DANN automatisieren.",
        "manufacturing": "Industry context Manufacturing: Machine downtime has direct cost consequences. Standardize first, THEN automate.",
        "e-commerce": "Branchenkontext E-Commerce: Skalierbarkeit zentral. Neue Shop-Systeme brauchen intensive Change-Kommunikation.",
        "ecommerce": "Industry context E-Commerce: Scalability is key. New shop systems require intensive change communication.",
        "pharma": "Branchenkontext Pharma: GMP-Compliance first. Schulung ist regulatorisch gefordert.",
        "pharmaceutical": "Industry context Pharma: GMP compliance first. Training is a regulatory requirement.",
        "retail": "Branchenkontext Einzelhandel: POS-Systeme scheitern an fehlender Schulung. Erst Schulung, DANN Rollout.",
        "handel": "Branchenkontext Handel: POS-Systeme scheitern an fehlender Schulung. Erst Schulung, DANN Rollout.",
        "einzelhandel": "Branchenkontext Einzelhandel: POS-Systeme scheitern an fehlender Schulung. Erst Schulung, DANN Rollout.",
        "saas": "Branchenkontext SaaS: API-Stabilität existenziell. Agile Prozesse brauchen klare Rollen.",
        "tech": "Industry context Tech: API stability is existential. Agile processes need clear roles.",
        "software": "Industry context Software: Feature adoption drives success. Pilot groups for new tools.",
    }

    industry_hint = ""
    industry_lower = industry.lower().strip() if industry else ""
    for key, hint in industry_hints.items():
        if key in industry_lower:
            industry_hint = hint
            break

    prompt = f"""
You are an operations transformation advisor writing for senior business decision-makers.

Your task is to convert a deterministic maturity assessment into a sharp, practical, structured advisory response.

Important:
- stay fully grounded in the provided data
- do not invent facts
- do not contradict the deterministic result
- do not use generic consulting language

Avoid phrases like:
- "it is important to note"
- "it is essential"
- "to improve efficiency"
- "to support growth"
- "to ensure long-term success"
- "to unlock potential"

Write like a strong operator or transformation lead:
- direct
- concrete
- business-oriented
- operational
- specific

Return exactly one valid JSON object with these keys:
- summary
- top_priorities
- lever
- risk
- next_step
- rag_references

Field logic:
- summary = 2-3 sentences, direct and honest
- top_priorities = 1 to 3 concrete priorities ranked by business leverage
- lever = the one dimension with the strongest system-wide impact
- risk = the consequence of inaction in operational terms
- next_step = one realistic first move
- rag_references = short plain labels derived directly from the retrieved context themes

Critical writing rules:
- The summary must name the weakest dimensions explicitly
- The summary must explain why transition is currently blocked
- Do not sound like a generic executive summary
- Every priority must describe a real managerial action
- Every rationale must explain why this action matters now
- Priorities should be derived primarily from required capacities, not generic management language
- Use the actual scores where useful
- Explain what breaks operationally
- Do not mention "JSON", "schema", or internal system design
- Do not use raw indicator codes like T2, P2, or R2 as action titles
- Do not write vague actions like "improve process maturity"
- If technology is weak, explain what that disrupts in operations
- If process is weak, explain where execution becomes unstable or error-prone
- If responsibility is weak, explain where follow-through or accountability breaks
- If adoption is weak, explain where workarounds undermine execution

Deterministic assessment context:
- Dimension scores: {json.dumps(scores)}
- Maturity descriptions: {json.dumps(maturity_descriptions)}
- Overall readiness: {overall}
- Bottlenecks: {json.dumps(bottlenecks)}
- Transition feasible: {feasible}
- Transition risk: {risk}
- Required changes: {json.dumps(required_changes)}
- Required capacities: {json.dumps(required_capacities)}
- Bottleneck details: {json.dumps(bottleneck_details)}
- Strengths: {json.dumps(strengths)}
- Weaknesses: {json.dumps(weaknesses)}
- Cross-dimension insights: {json.dumps(cross_dimension_insights)}
- Deterministic executive summary: {executive_summary}

Dimension score detail:
- Technology: {scores.get("T", 0)}
- Process: {scores.get("P", 0)}
- Responsibility: {scores.get("R", 0)}
- Adoption: {scores.get("A", 0)}
{f'- Industry context: {industry_hint}' if industry_hint else ''}

Retrieved knowledge:
{context_block}
"""

    return prompt.strip()


def build_compare_prompt(comparison: Dict[str, Any]) -> str:
    """
    Build the user prompt for comparative assessment explanation.

    Args:
        comparison: Comparison dictionary containing deltas, bottlenecks,
            and feasibility states.

    Returns:
        Prompt string for the comparison explanation endpoint.
    """
    t_delta = comparison.get("t_delta", 0)
    a_delta = comparison.get("a_delta", 0)
    risk_active = comparison.get("risk_alert", False)
    gap = comparison.get("critical_gap", 0)

    prompt = f"""
You are a system analyst writing for business decision-makers.

Your task is to explain structural differences between two system states.
Be concrete, operational, and specific.

Do not use generic phrases such as:
- "performance improved"
- "technology is better"
- "the process is more efficient"

Instead explain:
- what changed structurally
- what this changes in day-to-day operations
- why this matters for transition feasibility

Return valid JSON with:
- summary
- main_improvements
- transition_impact

Rules:
- summary = one concise sentence
- main_improvements = up to 3 short statements
- transition_impact = up to 3 short statements
- every statement must refer to a real structural change
- do not just repeat the raw deltas

Comparison context:
- R delta: {comparison["r_delta"]}
- P delta: {comparison["p_delta"]}
- T delta: {comparison["t_delta"]}
- A delta: {comparison["a_delta"]}
- Overall delta: {comparison["overall_readiness_delta"]}
- Transition feasible A: {comparison["transition_feasible_a"]}
- Transition feasible B: {comparison["transition_feasible_b"]}
- Bottlenecks A: {json.dumps(comparison["bottleneck_a"])}
- Bottlenecks B: {json.dumps(comparison["bottleneck_b"])}
"""

    if risk_active:
        prompt += f"""
    CRITICAL RISK DETECTED:
    The 'risk_alert' flag is TRUE. The technological advancement (T-delta: {t_delta}) 
    has significantly outpaced organizational adoption (A-delta: {a_delta}).
    The current gap is {gap}.

    In your summary and transition_impact:
    - Explicitly warn about the 'Digital Adoption Gap'.
    - Mention the risk of 'Shelfware' (expensive technology that stays unused).
    - Explain that Scenario B might fail despite better tech if training/acceptance isn't prioritized.
    """
    else:
        prompt += "\nAnalysis: Focus on how the structural changes in Scenario B resolve the bottlenecks found in Scenario A."
    return prompt.strip()


# =============================================================================
# OpenAI generation functions
# =============================================================================

def generate_explanation_openai(
    api_key: str,
    engine_result: Dict[str, Any],
    retrieved_context: List[str],
    industry: str = "",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Generate structured advisory output for a single assessment.

    This output is intended to align with core.output.AdvisorOutput.

    Args:
        api_key: OpenAI API key.
        engine_result: Deterministic assessment result.
        retrieved_context: Context lines from the retrieval layer and/or
            prior conversation history.
        industry: Industry context for branch-specific recommendations.
        temperature: Temperature setting for the model (0.2 for focused, 0.8 for creative).

    Returns:
        Dictionary containing structured advisory fields plus model metadata.
    """
    client = OpenAI(api_key=api_key)
    prompt = build_prompt(engine_result, retrieved_context, industry)

    response = client.responses.create(
        model=ADVISOR_MODEL_NAME,
        input=[
            {
                "role": "developer",
                "content": (
                    "You generate structured advisory outputs for operational maturity assessments. "
                    "Be concrete, specific, and operational. "
                    "Avoid generic management language and empty consulting phrases. "
                    "Ground every recommendation in the provided bottlenecks, scores, and required capacities. "
                    "Do not invent facts. "
                    "Return only valid JSON that matches the requested schema. "
                    "IMPORTANT: Do NOT use markdown formatting in any text fields. Write in plain text without **bold**, *italic*, or any other markdown syntax."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "name": "advisor_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "top_priorities": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "rank": {"type": "integer"},
                                    "dimension": {"type": "string", "enum": ["T", "P", "R", "A"]},
                                    "action": {"type": "string"},
                                    "rationale": {"type": "string"},
                                    "timeframe": {"type": ["string", "null"]},
                                },
                                "required": ["rank", "dimension", "action", "rationale", "timeframe"],
                                "additionalProperties": False,
                            },
                        },
                        "lever": {
                            "type": "object",
                            "properties": {
                                "dimension": {"type": "string", "enum": ["T", "P", "R", "A"]},
                                "explanation": {"type": "string"},
                            },
                            "required": ["dimension", "explanation"],
                            "additionalProperties": False,
                        },
                        "risk": {
                            "type": "object",
                            "properties": {
                                "level": {"type": "string", "enum": ["low", "medium", "high"]},
                                "consequence": {"type": "string"},
                            },
                            "required": ["level", "consequence"],
                            "additionalProperties": False,
                        },
                        "next_step": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "owner": {"type": ["string", "null"]},
                                "by_when": {"type": ["string", "null"]},
                            },
                            "required": ["action", "owner", "by_when"],
                            "additionalProperties": False,
                        },
                        "rag_references": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["summary", "top_priorities", "lever", "risk", "next_step", "rag_references"],
                    "additionalProperties": False,
                },
            }
        },
    )

    parsed = _safe_json_loads(response.output_text)

    if not parsed:
        return {
            "summary": "The advisory output could not be parsed into a valid structured response.",
            "top_priorities": [
                {
                    "rank": 1,
                    "dimension": "P",
                    "action": "Review the generated advisory output manually.",
                    "rationale": "The model response could not be validated as structured JSON and should be checked before use.",
                    "timeframe": "immediate",
                }
            ],
            "lever": {
                "dimension": "P",
                "explanation": "Manual review is required because the structured advisory output could not be validated."
            },
            "risk": {
                "level": "medium",
                "consequence": "The deterministic system result exists, but the AI advisory layer could not be turned into a reliable structured response."
            },
            "next_step": {
                "action": "Inspect the raw model output and retry the generation.",
                "owner": "System administrator",
                "by_when": "today",
            },
            "rag_references": [],
            "model_name": ADVISOR_MODEL_NAME,
            "prompt_version": ADVISOR_PROMPT_VERSION,
        }

    parsed["model_name"] = ADVISOR_MODEL_NAME
    parsed["prompt_version"] = ADVISOR_PROMPT_VERSION
    return parsed


def build_compare_prompt(comparison: Dict[str, Any]) -> str:
    """
    Erstellt den strategischen Prompt basierend auf dem Delta-Payload.
    """
    t_delta = comparison.get("t_delta", 0)
    a_delta = comparison.get("a_delta", 0)
    risk_active = comparison.get("risk_alert", False)
    gap = comparison.get("critical_gap", 0)
    company_name = comparison.get("company_name", "the company")
    industry = comparison.get("industry", "")

    # Branchenkontext für die Analyse
    industry_context = ""
    if industry:
        industry_lower = industry.lower()
        if any(k in industry_lower for k in ["logistik", "logistics"]):
            industry_context = "Branchenkontext: Logistik - Automation erfordert intensive Schulung von Lagerpersonal und Fahrern. Neue Systeme ohne Training → Akzeptanz bricht ein."
        elif any(k in industry_lower for k in ["produktion", "manufacturing", "fertigung"]):
            industry_context = "Branchenkontext: Produktion - Technologie-Investitionen ohne Prozessreife → Qualitätsprobleme. Erst standardisieren, DANN automatisieren."
        elif any(k in industry_lower for k in ["e-commerce", "e-commerce", "online"]):
            industry_context = "Branchenkontext: E-Commerce - Neue Shop/ERP-Systeme brauchen Change-Kommunikation an Marketing- und Operations-Teams. Skalierbarkeit vorher testen."
        elif any(k in industry_lower for k in ["pharma", "chemie"]):
            industry_context = "Branchenkontext: Pharma - Jede Technologie-Änderung muss validiert und dokumentiert sein. Compliance-Anforderungen einplanen."
        elif any(k in industry_lower for k in ["retail", "handel", "einzelhandel"]):
            industry_context = "Branchenkontext: Einzelhandel - Neue POS/Kassen-Systeme scheitern an Schulungsmängeln. Erst Key-User trainieren, DANN Rollout."
        elif any(k in industry_lower for k in ["saas", "tech", "software"]):
            industry_context = "Branchenkontext: SaaS/Tech - Feature-Adoption bei Entwicklerteams kritisch. Pilotgruppen und Onboarding vor Voll-Rollout."
        else:
            industry_context = f"Branchenkontext: {industry}"

    # Basis-Informationen für die KI
    prompt = f"""
Compare Scenario A (Current) and Scenario B (Future) for {company_name} ({industry}) based on these structural deltas:
- Technology: {t_delta}
- Process: {comparison.get('p_delta', 0)}
- Responsibility: {comparison.get('r_delta', 0)}
- Adoption: {a_delta}
- Overall Readiness Delta: {comparison.get('overall_readiness_delta', 0)}

Feasibility Change: {comparison.get('transition_feasible_a')} -> {comparison.get('transition_feasible_b')}

{industry_context}
"""

    # Die spezifische Risiko-Anweisung hinzufügen
    if risk_active:
        prompt += f"""
CRITICAL WARNING: The 'risk_alert' is TRUE. 
Note that the technological advancement (T-Delta: {t_delta}) has significantly outpaced 
organizational adoption (A-Delta: {a_delta}) with a critical gap of {gap}.

In your summary, you MUST:
1. Warn the user that this creates a 'Digital Adoption Gap'.
2. Explain that this leads to 'Shelfware' (expensive, unused technology).
3. Advise that Scenario B will likely fail unless adoption measures are drastically increased.
4. Give specific, industry-relevant recommendations for closing this gap (e.g., "For {industry}: Schedule 2-week training bootcamp before go-live").
"""
    else:
        prompt += "\nAnalysis: Focus on how the structural changes improve the system flow and give industry-specific recommendations."

    return prompt

def generate_compare_explanation_openai(
    api_key: str,
    comparison: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate structured comparison explanation for two assessments.

    Args:
        api_key: OpenAI API key.
        comparison: Deterministic comparison dictionary.

    Returns:
        Dictionary containing summary, main improvements, transition impact,
        and model metadata.
    """
    client = OpenAI(api_key=api_key)
    prompt = build_compare_prompt(comparison)

    response = client.responses.create(
        model=COMPARE_MODEL_NAME,
        input=[
            {
                "role": "developer",
                "content": (
                    "You explain the business impact of structural system changes. "
                    "Be concrete, concise, and operational. "
                    "Avoid generic management language. "
                    "Do not invent data. "
                    "Return only valid JSON that matches the requested schema."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "comparison_explanation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "main_improvements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "transition_impact": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["summary", "main_improvements", "transition_impact"],
                    "additionalProperties": False,
                },
            }
        },
    )

    parsed = _safe_json_loads(response.output_text)

    if not parsed:
        return {
            "summary": "The comparison explanation could not be parsed.",
            "main_improvements": [],
            "transition_impact": [],
            "model_name": COMPARE_MODEL_NAME,
            "prompt_version": COMPARE_PROMPT_VERSION,
        }

    return {
        "summary": parsed.get("summary", ""),
        "main_improvements": parsed.get("main_improvements", []),
        "transition_impact": parsed.get("transition_impact", []),
        "model_name": COMPARE_MODEL_NAME,
        "prompt_version": COMPARE_PROMPT_VERSION,
    }

def generate_explanation_openai_with_temperature(
    api_key: str,
    engine_result: Dict[str, Any],
    retrieved_context: List[str],
    temperature: float,
) -> Dict[str, Any]:
    """
    Generate structured advisory output with an explicit temperature setting.

    Args:
        api_key: OpenAI API key.
        engine_result: Deterministic engine result.
        retrieved_context: Retrieved context lines.
        temperature: Sampling temperature for the model.

    Returns:
        Structured response dictionary plus metadata.
    """
    client = OpenAI(api_key=api_key)
    prompt = build_prompt(engine_result, retrieved_context)

    response = client.responses.create(
        model=ADVISOR_MODEL_NAME,
        temperature=temperature,
        input=[
            {
                "role": "developer",
                "content": (
                    "You generate structured advisory outputs for operational maturity assessments. "
                    "Be concrete, specific, and operational. "
                    "Avoid generic management language and empty consulting phrases. "
                    "Ground every recommendation in the provided bottlenecks, scores, and required capacities. "
                    "Do not invent facts. "
                    "Return only valid JSON that matches the requested schema."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "advisor_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "top_priorities": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "rank": {"type": "integer"},
                                    "dimension": {"type": "string", "enum": ["T", "P", "R", "A"]},
                                    "action": {"type": "string"},
                                    "rationale": {"type": "string"},
                                    "timeframe": {"type": ["string", "null"]},
                                },
                                "required": ["rank", "dimension", "action", "rationale", "timeframe"],
                                "additionalProperties": False,
                            },
                        },
                        "lever": {
                            "type": "object",
                            "properties": {
                                "dimension": {"type": "string", "enum": ["T", "P", "R", "A"]},
                                "explanation": {"type": "string"},
                            },
                            "required": ["dimension", "explanation"],
                            "additionalProperties": False,
                        },
                        "risk": {
                            "type": "object",
                            "properties": {
                                "level": {"type": "string", "enum": ["low", "medium", "high"]},
                                "consequence": {"type": "string"},
                            },
                            "required": ["level", "consequence"],
                            "additionalProperties": False,
                        },
                        "next_step": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "owner": {"type": ["string", "null"]},
                                "by_when": {"type": ["string", "null"]},
                            },
                            "required": ["action", "owner", "by_when"],
                            "additionalProperties": False,
                        },
                        "rag_references": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["summary", "top_priorities", "lever", "risk", "next_step", "rag_references"],
                    "additionalProperties": False,
                },
            }
        },
    )

    parsed = _safe_json_loads(response.output_text)

    if not parsed:
        return {
            "summary": "The advisory output could not be parsed.",
            "top_priorities": [],
            "lever": {"dimension": "P", "explanation": "Parsing failed."},
            "risk": {"level": "medium", "consequence": "The model response could not be validated."},
            "next_step": {"action": "Retry generation.", "owner": None, "by_when": None},
            "rag_references": [],
            "model_name": ADVISOR_MODEL_NAME,
            "prompt_version": ADVISOR_PROMPT_VERSION,
            "temperature": temperature,
        }

    parsed["model_name"] = ADVISOR_MODEL_NAME
    parsed["prompt_version"] = ADVISOR_PROMPT_VERSION
    parsed["temperature"] = temperature
    return parsed


def _generate_explanation_with_temperature(
    client: OpenAI,
    engine_result: Dict[str, Any],
    retrieved_context: List[str],
    temperature: float,
) -> Dict[str, Any]:
    """
    Internal helper to generate one explanation with a specific temperature.
    """
    prompt = build_prompt(engine_result, retrieved_context)

    response = client.responses.create(
        model="gpt-4o-mini",
        temperature=temperature,
        input=[
            {
                "role": "developer",
                "content": (
                    "You explain structural constraints in systems. "
                    "Use concrete business language. "
                    "Do not recommend actions. "
                    "Do not invent data."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "assessment_explanation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "diagnosis": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "operational_impact": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "capacity_gap": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["diagnosis", "operational_impact", "capacity_gap"],
                    "additionalProperties": False,
                },
            }
        },
    )

    try:
        parsed = json.loads(response.output_text)
    except Exception:
        parsed = {
            "diagnosis": ["The explanation could not be parsed."],
            "operational_impact": ["Structured model output was invalid."],
            "capacity_gap": [],
        }

    return {
        "diagnosis": parsed["diagnosis"],
        "operational_impact": parsed["operational_impact"],
        "capacity_gap": parsed["capacity_gap"],
        "temperature": temperature,
    }


def generate_temperature_comparison_openai(
    api_key: str,
    engine_result: Dict[str, Any],
    retrieved_context: List[str],
    temperature_a: float,
    temperature_b: float,
) -> Dict[str, Any]:
    """
    Generate two structured explanation outputs for the same assessment
    using two different temperature values, plus an AI-generated comparison summary.
    """
    client = OpenAI(api_key=api_key)

    output_a = _generate_explanation_with_temperature(
        client=client,
        engine_result=engine_result,
        retrieved_context=retrieved_context,
        temperature=temperature_a,
    )

    output_b = _generate_explanation_with_temperature(
        client=client,
        engine_result=engine_result,
        retrieved_context=retrieved_context,
        temperature=temperature_b,
    )

    summary = _generate_temperature_comparison_summary(
        client=client,
        output_a=output_a,
        output_b=output_b,
        temperature_a=temperature_a,
        temperature_b=temperature_b,
    )

    return {
        "output_a": output_a,
        "output_b": output_b,
        "ai_summary": summary,
        "model_name": "gpt-4o-mini",
        "prompt_version": "v1_temperature_comparison_schema",
    }


def _generate_temperature_comparison_summary(
    client,
    output_a: Dict[str, Any],
    output_b: Dict[str, Any],
    temperature_a: float,
    temperature_b: float,
) -> str:
    """
    Generate a comparative summary that explains the differences between
    low-temperature (focused) and high-temperature (creative) responses.
    """
    summary_prompt = f"""You are analyzing two AI-generated explanations for the same system assessment, 
generated with different temperature settings.

Low Temperature ({temperature_a}) Response:
- Summary: {output_a.get('summary', 'N/A')}
- Top Priorities: {', '.join([p.get('action', 'N/A') for p in output_a.get('top_priorities', [])[:2]])}
- Main Lever: {output_a.get('lever', {}).get('explanation', 'N/A')[:150] if output_a.get('lever') else 'N/A'}

High Temperature ({temperature_b}) Response:
- Summary: {output_b.get('summary', 'N/A')}
- Top Priorities: {', '.join([p.get('action', 'N/A') for p in output_b.get('top_priorities', [])[:2]])}
- Main Lever: {output_b.get('lever', {}).get('explanation', 'N/A')[:150] if output_b.get('lever') else 'N/A'}

Please provide a brief comparison (2-3 sentences) that highlights:
1. Key differences in how the two responses approach the analysis
2. Which temperature setting might be more suitable for operational decision-making

Keep your response concise and in the same language as the input (German if the context suggests German, otherwise English)."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing AI response differences."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception:
        return ""