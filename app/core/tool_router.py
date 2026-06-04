"""
app/core/tool_router.py
───────────────────────
Deterministic cancer type → MCP tool mapping.
No LLM involved — pure routing logic.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class CancerType(str, Enum):
    """Supported cancer types from the frontend form."""
    BREAST = "breast"
    CERVICAL = "cervical"
    HEAD_NECK = "head_neck"
    PROSTATE = "prostate"
    BLADDER = "bladder"
    TESTICULAR = "testicular"
    COLON = "colon"
    RECTUM = "rectum"
    ESOPHAGEAL = "esophageal"
    GASTRIC = "gastric"
    PANCREATIC = "pancreatic"
    ANAL = "anal"
    LYMPHOMA = "lymphoma"
    CNS = "cns"


# ──────────────────────────────────────────────────────────────────────────────
# Mapping: CancerType → MCP Tool Name
# ──────────────────────────────────────────────────────────────────────────────

TOOL_MAP: dict[CancerType, str] = {
    CancerType.BREAST: "breast_cancer",
    CancerType.CERVICAL: "cervix_cancer",
    CancerType.HEAD_NECK: "hnscc_decision",
    CancerType.PROSTATE: "gu_prostate",
    CancerType.BLADDER: "gu_bladder",
    CancerType.TESTICULAR: "gu_testicular",
    CancerType.COLON: "gi_cancer",
    CancerType.RECTUM: "gi_cancer",
    CancerType.ESOPHAGEAL: "gi_cancer",
    CancerType.GASTRIC: "gi_cancer",
    CancerType.PANCREATIC: "gi_cancer",
    CancerType.ANAL: "gi_cancer",
    CancerType.LYMPHOMA: "lymphoma",
    CancerType.CNS: "cns_tumor",
}


# ──────────────────────────────────────────────────────────────────────────────
# Tool Router
# ──────────────────────────────────────────────────────────────────────────────

def get_tool_name(cancer_type: str | CancerType) -> str:
    """
    Deterministically map cancer_type to MCP tool name.
    
    Args:
        cancer_type: string or CancerType enum value
    
    Returns:
        MCP tool name (e.g., "breast_cancer")
    
    Raises:
        ValueError: if cancer_type is not recognized
    """
    if isinstance(cancer_type, str):
        try:
            cancer_type = CancerType(cancer_type.lower())
        except ValueError:
            raise ValueError(
                f"Unknown cancer type: {cancer_type}. "
                f"Valid types: {[c.value for c in CancerType]}"
            )
    
    if cancer_type not in TOOL_MAP:
        raise ValueError(f"No tool mapping for cancer type: {cancer_type}")
    
    return TOOL_MAP[cancer_type]


def validate_cancer_type(cancer_type: str) -> CancerType:
    """
    Validate and normalize cancer_type from frontend.
    
    Args:
        cancer_type: user-provided string
    
    Returns:
        Validated CancerType enum
    
    Raises:
        ValueError: if invalid
    """
    try:
        return CancerType(cancer_type.lower())
    except ValueError:
        raise ValueError(
            f"Invalid cancer_type: '{cancer_type}'. "
            f"Must be one of: {', '.join(c.value for c in CancerType)}"
        )


def get_supported_cancer_types() -> list[str]:
    """Return all supported cancer types."""
    return [c.value for c in CancerType]