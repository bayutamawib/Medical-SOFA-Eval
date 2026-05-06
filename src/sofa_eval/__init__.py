"""
sofa-eval: Calibrated Abstention Evaluation for Clinical LLMs
=============================================================

A modular Python library for evaluating the clinical reasoning quality
of Large Language Models (LLMs) through SOFA Score extraction verification,
Cactus Signal routing compliance, and Calibrated Abstention scoring.

Reference: Vincent et al., Intensive Care Med 1996
Project  : Gemma-Sync — Calibrated Abstention in Small Language Models
"""

# ---------------------------------------------------------------------------
# utils.py — Text extraction & SOFA table parsing
# ---------------------------------------------------------------------------
from sofa_eval.utils import (
    # Functions
    extract_text,
    extract_boxed,
    check_sofa_not_applicable,
    parse_sofa_table,
    parse_numeric_sofa_value,
    extract_map_from_cv_string,
    # Constants
    COMPONENT_ALIASES,
)

# ---------------------------------------------------------------------------
# oracle.py — Deterministic clinical verification engine
# ---------------------------------------------------------------------------
from sofa_eval.oracle import (
    # Functions
    score_sofa_oracle,
    validate_cv_score,
    score_abstention_bonus,
    # Clinical constants
    SOFA_COMPONENTS,
    SOFA_SCORE_THRESHOLDS,
    VASOPRESSOR_KEYWORDS,
    # Scoring parameters
    HALLUCINATION_PENALTY,
    CACTUS_ESCALATE_TOKEN,
    CACTUS_LOCAL_TOKEN,
    ABSTENTION_BONUS,
)

# ---------------------------------------------------------------------------
# rewards.py — GRPO reward functions for clinical LLM alignment
# ---------------------------------------------------------------------------
from sofa_eval.rewards import (
    reward_correctness,
    reward_sofa_oracle,
    reward_format,
    reward_process_quality,
)

__version__ = "0.1.0"
__author__ = "Narendra Bayutama Wibisono"

__all__ = [
    # --- utils ---
    "extract_text",
    "extract_boxed",
    "check_sofa_not_applicable",
    "parse_sofa_table",
    "parse_numeric_sofa_value",
    "extract_map_from_cv_string",
    "COMPONENT_ALIASES",
    # --- oracle ---
    "score_sofa_oracle",
    "validate_cv_score",
    "score_abstention_bonus",
    "SOFA_COMPONENTS",
    "SOFA_SCORE_THRESHOLDS",
    "VASOPRESSOR_KEYWORDS",
    "HALLUCINATION_PENALTY",
    "CACTUS_ESCALATE_TOKEN",
    "CACTUS_LOCAL_TOKEN",
    "ABSTENTION_BONUS",
    # --- rewards ---
    "reward_correctness",
    "reward_sofa_oracle",
    "reward_format",
    "reward_process_quality",
]
