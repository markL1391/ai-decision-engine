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

def build_prompt(engine_result: Dict[str, Any], retrieved_context: List[str]) -> str:
    """
    Build the user prompt for structured advisory output.

    The model is asked to turn deterministic assessment data into a
    concrete, operational advisory response that can be rendered in the UI.

    Args:
        engine_result: Deterministic engine output.
        retrieved_context: Context lines from the retrieval layer and/or
            conversation history.

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

    context_block = _build_context_block(retrieved_context)

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
    return prompt.strip()


# =============================================================================
# OpenAI generation functions
# =============================================================================

def generate_explanation_openai(
    api_key: str,
    engine_result: Dict[str, Any],
    retrieved_context: List[str],
) -> Dict[str, Any]:
    """
    Generate structured advisory output for a single assessment.

    This output is intended to align with core.output.AdvisorOutput.

    Args:
        api_key: OpenAI API key.
        engine_result: Deterministic assessment result.
        retrieved_context: Context lines from the retrieval layer and/or
            prior conversation history.

    Returns:
        Dictionary containing structured advisory fields plus model metadata.
    """
    client = OpenAI(api_key=api_key)
    prompt = build_prompt(engine_result, retrieved_context)

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