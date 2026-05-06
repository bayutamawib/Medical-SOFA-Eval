"""
sofa-eval: Calibrated Abstention Evaluation for Clinical LLMs
=============================================================

A modular Python library for evaluating the clinical reasoning quality
of Large Language Models (LLMs) through SOFA Score extraction verification,
Cactus Signal routing compliance, and Calibrated Abstention scoring.

Reference: Vincent et al., Intensive Care Med 1996
Project  : Gemma-Sync — Calibrated Abstention in Small Language Models
"""

from sofa_eval.utils import parse_sofa_table, extract_boxed, extract_text
from sofa_eval.oracle import (
    score_sofa_oracle,
    validate_cv_score,
    SOFA_COMPONENTS,
    SOFA_SCORE_THRESHOLDS,
    VASOPRESSOR_KEYWORDS,
)
from sofa_eval.rewards import (
    reward_correctness,
    reward_sofa_oracle,
    reward_format,
    reward_process_quality,
)

__version__ = "0.1.0"
__author__ = "Narendra Bayutama Wibisono"

__all__ = [
    "parse_sofa_table",
    "extract_boxed",
    "extract_text",
    "score_sofa_oracle",
    "validate_cv_score",
    "reward_correctness",
    "reward_sofa_oracle",
    "reward_format",
    "reward_process_quality",
    "SOFA_COMPONENTS",
    "SOFA_SCORE_THRESHOLDS",
    "VASOPRESSOR_KEYWORDS",
]
