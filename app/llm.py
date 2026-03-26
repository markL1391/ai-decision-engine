import json
from typing import Any, Dict, List

def build_prompt(engine_result: Dict[str, Any], retrieved_context: List[str]) -> str:
    scores = engine_result["dimension_scores"]
    bottlenecks = engine_result["bottlenecks"]
    overall = engine_result["overall_readiness"]
    feasible = engine_result["transition_feasible"]
    risk = engine_result["transition_risk"]
    required_changes = engine_result["required_changes"]

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

Retrieved knowledge:
{context_block}
"""
    return prompt.strip()

def generate_explanation_mock(engine_result: Dict[str, Any], retrieved_context: List[str]) -> Dict[str, Any]:
    """
    Mock version for MVP before real LLM integration.
    """
    bottlenecks = engine_result["bottlenecks"]

    why_limit = []
    blocks_transition = []

    if "A" in bottlenecks:
        why_limit.extend([
            "Low acceptance indictaes resistance to process changes.",
            "Digital tools are not consistently adopted across the system.",
            "Organizational readiness for change is insufficient.",
        ])
        blocks_transition.extend([
            "Employees do not trust new workflows sufficiently.",
            "Training and communication structures are not fully established."
            "The system lacks stable behavioural adoption",
        ])

    if "T" in bottlenecks:
        why_limit.extend([
            "Technology maturity is too low to support the target state.",
            "System integration remains insufficient.",
            "Automation capacity it not yet structurally embedded."
        ])
        blocks_transition.extend([
            "Low system availability increases operational fragility.",
            "Manual interventions remain too frequent.",
            "Technical infrastructure does not support scalable execution.",
        ])

    if not why_limit:
        why_limit = [
            "The system is constrained by low-scoring structural dimensions.",
            "The current maturity profile does not support the target level.",
            "The limiting dimension defines the current system boundary."
        ]
        blocks_transition = [
            "One or more dimensions remain below the target level.",
            "Structural readiness is not yet consistent across the system.",
            "The transition is blocked by unresolved maturity gaps.",
        ]

    return {
        "why_limit": why_limit[:3],
        "blocks_transition": blocks_transition[:3],
        "references": ["scale_defintion#1", "transition_logic#1"],
        "model_name": "mock-llm",
        "prompt_version": "v1",
    }