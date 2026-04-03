"""
rag.py

Lightweight retrieval layer for the Explainable AI Maturity Assessment System.

This module provides contextual knowledge snippets based on detected bottlenecks.
It is used to enrich LLM prompts with domain-specific explanations.

Note:
This is a controlled internal retrieval approach for the MVP.
It follows the principle of retrieval-augmented generation, but uses a static
knowledge base instead of embeddings or vector search.
"""

from typing import Any, Dict, List


# =============================================================================
# Internal knowledge base
# =============================================================================

KNOWLEDGE_BASE = {
    "R": [
        "Responsibility defines the clarity of ownership and decision authority.",
        "A structural responsibility bottleneck causes processes to become unstable under pressure when ownership is unclear.",
        "Without clear accountability at level 2 or above, scalable execution lacks a stable organisational basis.",
    ],
    "P": [
        "Process maturity reflects the standardisation and repeatability of workflows.",
        "A structural process bottleneck causes error rates to rise significantly when workload increases.",
        "Weak process maturity can make technology investments ineffective because unstable execution undermines system benefits.",
    ],
    "T": [
        "Technology represents system support, integration, and automation capability.",
        "A structural technology bottleneck limits reliable data flow when interfaces are missing or systems remain fragmented.",
        "Low technology maturity acts as a practical barrier to scaling process execution and role clarity.",
    ],
    "A": [
        "Adoption describes the organisation’s willingness to use new tools and ways of working.",
        "A structural adoption bottleneck leads employees to rely on shadow processes outside the intended system.",
        "Low adoption undermines scale because manual workarounds weaken process discipline and system consistency.",
    ],
}


# =============================================================================
# Retrieval functions
# =============================================================================

def retrieve_context(engine_result: Dict[str, Any]) -> List[str]:
    """
    Retrieve context snippets based on detected bottlenecks.

    The function selects only those knowledge entries that match the current
    bottleneck dimensions identified by the deterministic engine.

    Args:
        engine_result: Deterministic engine output containing bottlenecks.

    Returns:
        List of context strings relevant to the current assessment.
    """
    bottlenecks = engine_result.get("bottlenecks", [])
    context: List[str] = []

    for bottleneck in bottlenecks:
        knowledge_entries = KNOWLEDGE_BASE.get(bottleneck, [])
        context.extend(knowledge_entries)

    if not context:
        context.append("The system currently shows balanced maturity across all dimensions.")
        context.append("Future transition success depends on maintaining the current operational standard.")

    return context


def get_dimension_name(dim_key: str) -> str:
    """
    Convert a dimension key into a human-readable label.

    Args:
        dim_key: One of 'R', 'P', 'T', or 'A'.

    Returns:
        Human-readable dimension name.
    """
    names = {
        "R": "Responsibility",
        "P": "Process",
        "T": "Technology",
        "A": "Adoption",
    }
    return names.get(dim_key, dim_key)