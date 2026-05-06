"""
sofa_eval.rewards — GRPO Reward Functions for Clinical LLM Alignment
=====================================================================

Four-tier reward architecture for Reinforcement Learning from
Verifiable Rewards (RLVR) via Group Relative Policy Optimization (GRPO).

Reward Architecture:
  R = 0.50 × correctness     — RLVR exact-match \\boxed{}
    + 0.20 × sofa_oracle     — SOFA 6-system verifier
    + 0.10 × format          — LaTeX structure compliance
    + 0.20 × process         — CoT quality + metacognitive calibration
  [+0.20 bonus]              — Calibrated Abstention (N/P detection)
"""

import re
from typing import Any, List

from sofa_eval.oracle import (
    CACTUS_ESCALATE_TOKEN,
    CACTUS_LOCAL_TOKEN,
    score_abstention_bonus,
    score_sofa_oracle,
)
from sofa_eval.utils import (
    check_sofa_not_applicable,
    extract_boxed,
    extract_text,
)


# ---------------------------------------------------------------------------
# Reward 1: Correctness (RLVR) — weight 0.50
# ---------------------------------------------------------------------------

def reward_correctness(
    prompts: List[Any],
    completions: List[Any],
    answer: List[str],
    **kwargs: Any,
) -> List[float]:
    r"""RLVR hard reward: binary exact-match of ``\boxed{X}`` against ground-truth.

    This is the dominant signal (weight 0.50). The model receives 1.0 only
    if its extracted answer letter matches the gold-standard USMLE answer.

    Args:
        prompts: List of input prompts (unused but required by GRPO API).
        completions: List of model-generated completions.
        answer: List of ground-truth answer letters (e.g., ``["A", "C", ...]``).

    Returns:
        List of floats: 1.0 (correct) or 0.0 (incorrect/absent) per sample.
    """
    rewards: List[float] = []
    for completion, expected in zip(completions, answer):
        text = extract_text(completion)
        extracted = extract_boxed(text)
        if extracted and extracted == expected.strip().upper():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward 2: SOFA Oracle + Calibrated Abstention — weight 0.20 (+0.20 bonus)
# ---------------------------------------------------------------------------

def reward_sofa_oracle(
    prompts: List[Any],
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    """SOFA Oracle reward (weight 0.20) + Calibrated Abstention bonus (up to +0.20).

    Calls the deterministic SOFA verification engine and adds the
    abstention bonus for appropriate uncertainty signaling.

    Args:
        prompts: List of input prompts.
        completions: List of model-generated completions.

    Returns:
        List of floats in [0.0, 1.2] per sample.
    """
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        text = extract_text(completion)
        prompt_text = extract_text(prompt) if prompt is not None else ""
        base = score_sofa_oracle(text)
        bonus = score_abstention_bonus(text, prompt_text)
        combined = min(base + bonus, 1.2)
        rewards.append(round(combined, 4))
    return rewards


# ---------------------------------------------------------------------------
# Reward 3: Format Compliance — weight 0.10
# ---------------------------------------------------------------------------

def reward_format(
    prompts: List[Any],
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    r"""Validates the structural output format (weight 0.10).

    Checks for:
    - Valid ``\boxed{X}`` answer (0.40)
    - SOFA table presence (0.30)
    - Step-by-step reasoning markers (0.20)
    - Confidence/N/P metadata labels (0.10)

    Args:
        prompts: List of input prompts.
        completions: List of model-generated completions.

    Returns:
        List of floats in [0.0, 1.0] per sample.
    """
    rewards: List[float] = []
    for completion in completions:
        text = extract_text(completion)
        score = 0.0

        # Boxed answer
        boxed = extract_boxed(text)
        if boxed and boxed in ("A", "B", "C", "D"):
            score += 0.40
        elif re.search(r"\\boxed\{", text):
            score += 0.15

        # SOFA table presence
        has_table = bool(re.search(
            r"\|\s*(?:Respiratory|Coagulation|Liver|Cardiovascular|CNS|Neurological|Renal)",
            text, re.IGNORECASE,
        ))
        if has_table or check_sofa_not_applicable(text):
            score += 0.30

        # Step markers
        steps_found = sum(1 for p in [
            r"Step\s*1|Clinical Data Extraction|SOFA Assessment",
            r"Step\s*2|Clinical Reasoning|Differential",
            r"Step\s*3|Uncertainty|Metacognitive",
            r"Step\s*4|Final Answer",
        ] if re.search(p, text, re.IGNORECASE))
        score += 0.20 * (steps_found / 4)

        # Confidence / N/P metadata
        if re.search(
            r"(?:high|moderate|low)\s+confidence|N/P",
            text, re.IGNORECASE,
        ):
            score += 0.10

        rewards.append(round(min(score, 1.0), 4))
    return rewards


# ---------------------------------------------------------------------------
# Reward 4: Process Quality (CoT Heuristic) — weight 0.20
# ---------------------------------------------------------------------------

def reward_process_quality(
    prompts: List[Any],
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    """Offline Chain-of-Thought quality heuristic (weight 0.20).

    Evaluates the depth and quality of clinical reasoning by checking:
    - Medical terminology usage (pathophysiology, differential, etc.)
    - Response length (optimal: 300-800 words)
    - Causal reasoning connectives
    - Differential elimination language
    - Logical transition words
    - Cactus routing token presence

    Args:
        prompts: List of input prompts.
        completions: List of model-generated completions.

    Returns:
        List of floats in [0.0, 1.0] per sample.
    """
    rewards: List[float] = []
    for completion in completions:
        text = extract_text(completion)
        score = 0.0
        max_score = 6.0

        # Medical terminology
        medical_terms = [
            r"pathophysiology", r"differential\s+diagnosis", r"etiology",
            r"mechanism\s+of\s+action", r"clinical\s+presentation", r"prognosis",
            r"(?:renal|hepatic|pulmonary|cardiac)\s+(?:failure|insufficiency)",
            r"(?:sepsis|ARDS|DIC|AKI|SIRS)",
            r"(?:PaO2|FiO2|platelet|bilirubin|creatinine|GCS|MAP)\b",
            r"vasopressor", r"Glasgow\s+Coma\s+Scale",
            r"first.?line\s+(?:treatment|therapy)",
        ]
        hits = sum(
            1 for t in medical_terms if re.search(t, text, re.IGNORECASE)
        )
        score += min(hits / 5.0, 1.0)

        # Response length
        words = len(text.split())
        if 300 <= words <= 800:
            score += 1.0
        elif 150 <= words < 300 or 800 < words <= 1200:
            score += 0.6
        elif words >= 50:
            score += 0.3

        # Causal reasoning
        causal = [
            r"because|since|given\s+that|as\s+evidenced\s+by",
            r"consistent\s+with|compatible\s+with|most\s+likely",
        ]
        score += min(
            sum(1 for p in causal if re.search(p, text, re.IGNORECASE)) / 2.0,
            1.0,
        )

        # Differential elimination
        elim = [
            r"option\s+[ABCD]\s+is\s+(?:incorrect|unlikely|wrong)",
            r"ruled?\s+out|less\s+likely\s+because|not\s+consistent\s+with",
        ]
        score += min(
            sum(1 for p in elim if re.search(p, text, re.IGNORECASE)) / 2.0,
            1.0,
        )

        # Logical transitions
        transitions = [
            r"\b(?:therefore|thus|hence|consequently)\b",
            r"\b(?:however|although|despite|nevertheless)\b",
            r"\b(?:furthermore|additionally|moreover)\b",
        ]
        score += min(
            sum(
                1 for p in transitions if re.search(p, text, re.IGNORECASE)
            ) / 2.0,
            1.0,
        )

        # Cactus routing token (exactly one = best)
        has_escalate = CACTUS_ESCALATE_TOKEN in text
        has_local_ok = CACTUS_LOCAL_TOKEN in text
        if has_escalate ^ has_local_ok:
            score += 1.0
        elif has_escalate or has_local_ok:
            score += 0.5

        rewards.append(round(min(score / max_score, 1.0), 4))
    return rewards
