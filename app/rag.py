from typing import Any, Dict, List

KNOWLEDGE_BASE = {
    "A": [
        "Acceptance refers to the organisational willingness to adopt new processes and tools."
        "Low acceptance often limits system transitions even when technology is available.",
    ],
    "P": [
        "Process maturity reflects how structured, repeatable, and stable workflows are.",
        "Weak process maturiy reduces transition feasibility and operational consistency.",
    ],
    "T": [
        "Technology represents system support, integration, and automation capability.",
        "Low technology maturity can block scalability and automation transitions.",
    ],
    "R": [
        "Responsibility reflects clarity of ownership, accountability, and role assignment.",
        "Unclear responsibilities may undermine structural readiness.",
    ],
}

def retrieve_context(engine_result: Dict[str, Any]) -> List[str]:
    bottlenecks = engine_result.get("bottlenecks", [])
    context: List[str] = []

    for bottleneck in bottlenecks:
        context.extend(KNOWLEDGE_BASE.get(bottleneck, []))

    if not context:
        context.extend("Structural system transitions depend on maturity across all dimensions.")

    return context[:5]