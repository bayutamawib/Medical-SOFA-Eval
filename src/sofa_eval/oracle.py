"""
sofa_eval.oracle — SOFA Score Verification Engine
==================================================

Deterministic clinical verification of LLM-generated SOFA tables.
Validates extracted organ-system scores against established clinical
thresholds (Vincent et al., Intensive Care Med 1996).

Scoring breakdown:
  (0.25) table presence  + (0.20) coverage  + (0.20) validity
  (0.20) plausibility    + (0.15) arithmetic
  Penalty: -HALLUCINATION_PENALTY per fabricated score.
"""

import re
from typing import Any, Callable, Dict, Optional

from sofa_eval.utils import (
    check_sofa_not_applicable,
    extract_map_from_cv_string,
    parse_numeric_sofa_value,
    parse_sofa_table,
)


# ---------------------------------------------------------------------------
# SOFA Clinical Constants
# ---------------------------------------------------------------------------

SOFA_COMPONENTS = [
    "Respiratory",      # PaO2/FiO2 ratio
    "Coagulation",      # Platelet count (×10³/μL)
    "Liver",            # Bilirubin (mg/dL)
    "Cardiovascular",   # MAP / Vasopressors
    "CNS",              # Glasgow Coma Scale
    "Renal",            # Creatinine (mg/dL)
]

SOFA_SCORE_THRESHOLDS: Dict[str, Dict[int, Callable[[float], bool]]] = {
    "Respiratory": {
        0: lambda v: v >= 400,
        1: lambda v: 300 <= v < 400,
        2: lambda v: 200 <= v < 300,
        3: lambda v: 100 <= v < 200,
        4: lambda v: v < 100,
    },
    "Coagulation": {
        0: lambda v: v >= 150,
        1: lambda v: 100 <= v < 150,
        2: lambda v: 50 <= v < 100,
        3: lambda v: 20 <= v < 50,
        4: lambda v: v < 20,
    },
    "Liver": {
        0: lambda v: v < 1.2,
        1: lambda v: 1.2 <= v < 2.0,
        2: lambda v: 2.0 <= v < 6.0,
        3: lambda v: 6.0 <= v < 12.0,
        4: lambda v: v >= 12.0,
    },
    "CNS": {
        0: lambda v: v == 15,
        1: lambda v: 13 <= v <= 14,
        2: lambda v: 10 <= v <= 12,
        3: lambda v: 6 <= v <= 9,
        4: lambda v: v < 6,
    },
    "Renal": {
        0: lambda v: v < 1.2,
        1: lambda v: 1.2 <= v < 2.0,
        2: lambda v: 2.0 <= v < 3.5,
        3: lambda v: 3.5 <= v < 5.0,
        4: lambda v: v >= 5.0,
    },
    "Cardiovascular": {
        0: lambda v: v >= 70,
        1: lambda v: v < 70,
    },
}

VASOPRESSOR_KEYWORDS = [
    "dopamine", "dobutamine", "epinephrine", "norepinephrine",
    "vasopressin", "phenylephrine", "milrinone",
]

HALLUCINATION_PENALTY = 0.10

CACTUS_ESCALATE_TOKEN = "<|escalate|>"
CACTUS_LOCAL_TOKEN = "<|local_ok|>"
ABSTENTION_BONUS = 0.20


# ---------------------------------------------------------------------------
# Cardiovascular Score Validation
# ---------------------------------------------------------------------------

def validate_cv_score(value_str: str, claimed_score: int) -> Optional[bool]:
    """Validate a Cardiovascular SOFA sub-score against MAP + vasopressor evidence.

    Checks whether the model's claimed cardiovascular score is consistent
    with the blood pressure values and vasopressor mentions in the cell.

    Args:
        value_str: The raw cardiovascular value string from the SOFA table.
        claimed_score: The integer score (0-4) the model assigned.

    Returns:
        True if plausible, False if implausible, None if indeterminate.
    """
    value_lower = value_str.lower()
    map_val = extract_map_from_cv_string(value_lower)
    has_vasopressors = any(vp in value_lower for vp in VASOPRESSOR_KEYWORDS)

    if claimed_score == 0:
        if map_val is not None:
            return map_val >= 70 and not has_vasopressors
        if any(kw in value_lower for kw in ("stable", "normal", "assumed")):
            return True
        return None
    elif claimed_score == 1:
        if map_val is not None:
            return map_val < 70
        return None
    elif claimed_score in (2, 3, 4):
        if has_vasopressors:
            return True
        if map_val is not None and map_val < 70:
            return True
        return False
    return None


# ---------------------------------------------------------------------------
# Core Oracle Scoring Function
# ---------------------------------------------------------------------------

def score_sofa_oracle(text: str) -> float:
    """Precision SOFA Oracle — deterministic clinical verification.

    Evaluates the quality of a model-generated SOFA table by checking:
    1. Table presence (0.25)
    2. Coverage of 6 organ systems (0.20)
    3. Score validity — each score is 0-4 or N/P (0.20)
    4. Clinical plausibility — scores match physiological data (0.20)
    5. Arithmetic — total score matches sum of sub-scores (0.15)

    A hallucination penalty is applied for each fabricated value.

    Args:
        text: The full model output text.

    Returns:
        Float score in [0.0, 1.0].
    """
    if check_sofa_not_applicable(text):
        return 1.0

    components = parse_sofa_table(text)
    if not components:
        return 0.0

    score = 0.0
    hallucination_count = 0

    # (1) Table presence bonus
    score += 0.25

    # (2) Coverage: how many of the 6 canonical components are present
    found = set(components.keys()) & set(SOFA_COMPONENTS)
    score += 0.20 * (len(found) / len(SOFA_COMPONENTS))

    # (3) Validity: each score cell contains a valid value
    valid_scores = 0
    for comp_data in components.values():
        s = comp_data["score"].strip()
        if s in ("0", "1", "2", "3", "4") or s.upper() in ("N/P", "NOT PROVIDED"):
            valid_scores += 1
    score += 0.20 * (valid_scores / max(len(components), 1))

    # (4) Plausibility: cross-reference values against clinical thresholds
    plausible_count = 0
    checked_count = 0

    for comp_name, comp_data in components.items():
        raw_value = comp_data["value"].strip()
        raw_score = comp_data["score"].strip()

        is_np = raw_value.upper() in ("N/P", "NOT PROVIDED", "N/ P", "")
        if is_np:
            continue

        if raw_score in ("0", "1", "2", "3", "4"):
            has_numeric_content = bool(re.search(r"\d", raw_value))
            has_vasopressor = any(
                vp in raw_value.lower() for vp in VASOPRESSOR_KEYWORDS
            )
            has_stability_kw = any(
                kw in raw_value.lower()
                for kw in ("stable", "normal", "assumed")
            )
            if not (has_numeric_content or has_vasopressor or has_stability_kw):
                hallucination_count += 1
                continue

        # Cardiovascular — special validation via MAP + vasopressors
        if comp_name == "Cardiovascular" and raw_score in (
            "0", "1", "2", "3", "4"
        ):
            checked_count += 1
            cv_valid = validate_cv_score(raw_value, int(raw_score))
            if cv_valid is True:
                plausible_count += 1
            continue

        if raw_score not in ("0", "1", "2", "3", "4"):
            continue

        claimed = int(raw_score)
        numeric_val = parse_numeric_sofa_value(raw_value)
        if numeric_val is None:
            continue

        checked_count += 1
        threshold_key = comp_name if comp_name in SOFA_SCORE_THRESHOLDS else None
        if threshold_key and claimed in SOFA_SCORE_THRESHOLDS[threshold_key]:
            if SOFA_SCORE_THRESHOLDS[threshold_key][claimed](numeric_val):
                plausible_count += 1

    if checked_count > 0:
        score += 0.20 * (plausible_count / checked_count)
    else:
        score += 0.10

    # (5) Arithmetic: verify total SOFA score matches sum of sub-scores
    total_match = re.search(
        r"\*\*Total SOFA Score:\*\*\s*(\d+)\s*/\s*(\d+)", text, re.IGNORECASE,
    )
    if total_match:
        claimed_total = int(total_match.group(1))
        numeric_subs = [
            int(d["score"].strip())
            for d in components.values()
            if d["score"].strip() in ("0", "1", "2", "3", "4")
        ]
        if numeric_subs:
            diff = abs(sum(numeric_subs) - claimed_total)
            score += 0.15 * max(0.0, 1.0 - diff * 0.25)
        else:
            score += 0.075

    # Apply hallucination penalty
    score -= hallucination_count * HALLUCINATION_PENALTY
    return max(min(score, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Calibrated Abstention Bonus
# ---------------------------------------------------------------------------

def score_abstention_bonus(text: str, prompt_text: str = "") -> float:
    """Calibrated Abstention Reward (+0.20 bonus) with N/P-guessing penalty.

    Counts abstention signals (N/P markers, SOFA_NOT_APPLICABLE,
    confidence labels, escalation tokens) and awards a bonus when
    sufficient uncertainty indicators are present.

    Args:
        text: The full model output text.
        prompt_text: The original prompt text (for context-aware checks).

    Returns:
        Float bonus: +0.20 if >= 2 abstention signals, 0.0 otherwise.
        May include -0.10 penalty for false-confidence routing.
    """
    abstention_signals = 0

    # Count N/P markers
    np_count = len(re.findall(r"\bN/P\b", text))
    if np_count >= 1:
        abstention_signals += 1
    if np_count >= 3:
        abstention_signals += 1

    # SOFA not applicable with reason
    if check_sofa_not_applicable(text):
        reason_present = bool(
            re.search(r"\*\*Reason:\*\*", text, re.IGNORECASE)
        )
        abstention_signals += 2 if reason_present else 1

    # Confidence labels
    if re.search(r"(?:high|moderate|low)\s+confidence", text, re.IGNORECASE):
        abstention_signals += 1

    # Assumed-value markers
    if re.search(r"\[assumed:", text, re.IGNORECASE):
        abstention_signals += 1

    # Cactus routing tokens
    has_escalate = CACTUS_ESCALATE_TOKEN in text
    has_local_ok = CACTUS_LOCAL_TOKEN in text
    if has_escalate:
        abstention_signals += 2

    # False-confidence penalty: model claims local_ok despite many N/P
    penalty = 0.0
    has_claimed_total = bool(re.search(
        r"\*\*Total SOFA Score:\*\*\s*\d+", text, re.IGNORECASE,
    ))
    if has_local_ok and np_count >= 3 and has_claimed_total:
        penalty = -0.10

    base_bonus = ABSTENTION_BONUS if abstention_signals >= 2 else 0.0
    return base_bonus + penalty
