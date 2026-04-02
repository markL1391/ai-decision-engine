import json
from typing import Any, Dict, List

from openai import OpenAI

def build_prompt(engine_result: Dict[str, Any], retrieved_context: List[str]) -> str:
    scores = engine_result["dimension_scores"]
    bottlenecks = engine_result["bottlenecks"]
    overall = engine_result["overall_readiness"]
    feasible = engine_result["transition_feasible"]
    risk = engine_result["transition_risk"]
    required_changes = engine_result["required_changes"]
    bottleneck_details = engine_result["bottleneck_details"]
    required_capacities = engine_result["required_capacities"]

    context_block = "\n".join(f"- {item}" for item in retrieved_context)

    prompt = f"""
You are a system analyst.

Your task is to explain structural limitations in a system.
Do not make recommendations.
Do not change any scores.
Do not invent missing data.

Return valid JSON with the following keys:
- why_limit
- blocks_transition
- references

Context:
- Dimension scores: {json.dumps(scores)}
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

def generate_explanation_openai(
        api_key: str,
        engine_result: Dict[str, Any],
        retrieved_context: List[str]
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    prompt = build_prompt(engine_result, retrieved_context)

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": (
                    "You explain structural constraints in systems. "
                    "Do not recommend actions. "
                    "Do not change any scores. "
                    "Do not invent data."
                    "Do not mention variable names like R, P, T, A directly."
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
                        "why_limit": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "blocks_transition": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "references": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["why_limit", "blocks_transition", "references"],
                    "additionalProperties": False,
                },
            },
        },
    )


    parsed = json.loads(response.output_text)

    return {
        "why_limit": parsed["why_limit"],
        "blocks_transition": parsed["blocks_transition"],
        "references": parsed["references"],
        "model_name": "gpt-5-mini",
        "prompt_version": "v2_json_schema",
    }

def build_compare_prompt(comparison: dict) -> str:
    prompt = f"""
You are a system analyst.

Your task is to explain structural differences between two system states.

Do not make recommendations.
Do not invent data.

Return valid JSON with:
- summary
- main_improvements
- transition_impact

Context:
- R delta: {comparison["r_delta"]}
- P delta: {comparison["p_delta"]}
- T delta: {comparison["t_delta"]}
- A delta: {comparison["a_delta"]}
- Overall delta: {comparison["overall_readiness_delta"]}
- Transition feasible A: {comparison["transition_feasible_a"]}
- Transition feasible B: {comparison["transition_feasible_b"]}
- Bottlenecks A: {json.dumps(comparison["bottleneck_a"])}
- Bottlenecks B: {json.dumps(comparison["bottleneck_b"])}

Rules:
- Use concrete language
- Avoid generic business buzzwords
- Be specific (e.g. coordination, stability, scalability)
- Focus on what changes in real operations
- If one state has no bottlenecks, state that clearly
- Map underlying changes to these areas:
  - Responsibility → ownership and accountability
  - Process → consistency and efficiency
  - Technology → system support and scalability
  - Acceptance → willingness to adopt changes
"""
    return prompt.strip()

def generate_compare_explanation_openai(
        api_key: str,
        comparison: Dict[str, Any],
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    prompt = build_compare_prompt(comparison)

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": (
                    "You explain strucutural differences systems for business decision-makers. "
                    "Do not recommend actions. "
                    "Do not invent data. "
                    "Keep the explanation concise and factual. "
                    "Use simple, concrete language. "
                    "Avoid abstract phrases like 'structural gains' or 'operational capacity'. "
                    "Describe what actually improves in practice. "
                    "Focus on real-world impact (e.g. coordination, stability, scalability). "
                    "Each list must contain MAX 3 items. "
                    "Focus on business impact, not numbers. "
                    "Avoid mentioning deltas or numeric values unless critical. "
                    "Do not mention variable names like R, P, T, A directly. "
                    "Each item must be ONE short sentence."
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
                    "properties":{
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

    try:
        parsed = json.loads(response.output_text)
    except Exception:
        return {
            "error": "Invalid JSON from model",
            "raw_output": response.output_text
        }

    return {
        "summary": parsed["summary"],
        "main_improvements": parsed["main_improvements"],
        "transition_impact": parsed["transition_impact"],
        "model_name": "gpt-5-mini",
        "prompt_version": "v1_compare_json_schema",
    }