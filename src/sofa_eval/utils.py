"""
sofa_eval.utils — Text Extraction and SOFA Table Parsing
========================================================

All regex logic for parsing LLM-generated Markdown SOFA tables
and extracting structured clinical data from model completions.
"""

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Component Alias Map (canonical mapping for organ system names)
# ---------------------------------------------------------------------------
COMPONENT_ALIASES: Dict[str, str] = {
    "respiratory": "Respiratory",
    "respiration": "Respiratory",
    "coagulation": "Coagulation",
    "coag": "Coagulation",
    "platelet": "Coagulation",
    "liver": "Liver",
    "hepatic": "Liver",
    "bilirubin": "Liver",
    "cardiovascular": "Cardiovascular",
    "cardio": "Cardiovascular",
    "map": "Cardiovascular",
    "neurological": "CNS",
    "neuro": "CNS",
    "cns": "CNS",
    "gcs": "CNS",
    "renal": "Renal",
    "kidney": "Renal",
    "creatinine": "Renal",
}

# Pre-compiled row pattern for Markdown SOFA tables.
# Uses lazy search ([^|]+?) and whitespace cleanup (\s*) in each cell.
_ROW_PATTERN = re.compile(
    r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
    re.MULTILINE | re.IGNORECASE,
)


def extract_text(completion: Any) -> str:
    """Extract plain text from a completion (str, list of dicts, or dict).

    Handles the various formats returned by HuggingFace generation pipelines:
    - Raw string
    - List of message dicts (chat format)
    - Single message dict

    Args:
        completion: The raw model output in any supported format.

    Returns:
        The extracted plain text string.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in completion
        )
    if isinstance(completion, dict):
        return completion.get("content", str(completion))
    return str(completion)


def extract_boxed(text: str) -> Optional[str]:
    r"""Extract the answer letter from LaTeX ``\boxed{X}`` notation.

    Handles variants: ``\boxed{A}``, ``$$\boxed{B}$$``, ``$\boxed{C}$``,
    and bare ``boxed{D}``.

    Args:
        text: The full model output text.

    Returns:
        The uppercase answer letter (A-D), or None if not found.
    """
    patterns = [
        r"\\boxed\{\s*([A-Da-d])\s*\}",
        r"\$+\\boxed\{\s*([A-Da-d])\s*\}\$+",
        r"boxed\{\s*([A-Da-d])\s*\}",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).upper()
    return None


def check_sofa_not_applicable(text: str) -> bool:
    """Detect ``SOFA_NOT_APPLICABLE`` flag for non-critical case abstention.

    Args:
        text: The full model output text.

    Returns:
        True if the model explicitly flagged SOFA as not applicable.
    """
    return bool(re.search(
        r"SOFA_NOT_APPLICABLE|SOFA\s+(?:is\s+)?not\s+applicable",
        text, re.IGNORECASE,
    ))


def parse_sofa_table(text: str) -> Dict[str, Dict[str, str]]:
    """Parse a Markdown SOFA table into structured component data.

    Extracts the six SOFA organ-system rows from a model-generated
    Markdown table. Handles CNS/Neurological as synonyms per the
    clinical protocol.

    The expected table format is::

        | SOFA Component | Parameter | Value | Score |
        |----------------|-----------|-------|-------|
        | Respiratory    | PaO2/FiO2| 350   | 1     |

    Args:
        text: The full model output text containing a Markdown table.

    Returns:
        Dict mapping canonical component name to
        ``{"parameter": str, "value": str, "score": str}``.
    """
    result: Dict[str, Dict[str, str]] = {}
    for m in _ROW_PATTERN.findall(text):
        comp_raw, param, value, score = [x.strip() for x in m]

        # Skip header and separator rows
        if comp_raw.lower() in ("sofa component", "---", "") or \
                re.match(r"^[-:|\s]+$", comp_raw):
            continue

        # Canonical mapping using alias dictionary
        canonical = next(
            (v for k, v in COMPONENT_ALIASES.items() if k in comp_raw.lower()),
            None,
        )
        if canonical:
            result[canonical] = {
                "parameter": param,
                "value": value,
                "score": score,
            }

    return result


def parse_numeric_sofa_value(value_str: str) -> Optional[float]:
    """Extract a numeric value from a SOFA table cell for threshold validation.

    Cleans the cell string of units, brackets, and inequality symbols,
    then extracts the first numeric value.

    Args:
        value_str: The raw value string from the SOFA table cell.

    Returns:
        The extracted float value, or None if parsing fails.
    """
    cleaned = re.sub(r"\[assumed:\s*[^=]*=\s*", "", value_str)
    cleaned = re.sub(r"[a-zA-Z\u2082/\u03bc\u00b0%\]\[]+", "", cleaned)
    cleaned = re.sub(r"[><>=\u2265\u2264]", "", cleaned)
    cleaned = cleaned.strip()
    m = re.search(r"(\d+\.?\d*)", cleaned)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def extract_map_from_cv_string(value_str: str) -> Optional[float]:
    """Extract Mean Arterial Pressure (mmHg) from a cardiovascular value string.

    Handles BP string (e.g. ``120/80``), direct MAP label, and bare numeric.
    MAP = (SBP + 2 × DBP) / 3  [Vincent et al., 1996]

    Args:
        value_str: The cardiovascular value string from the SOFA table.

    Returns:
        Computed MAP in mmHg, or None if extraction fails.
    """
    text = value_str.strip().lower()

    # Try BP format (e.g., "BP: 120/80")
    bp_match = re.search(
        r"(?:bp|blood\s*pressure)?[:\s]*(?<!\.)\b(\d{2,3})\s*/\s*(\d{2,3})\b",
        text, re.IGNORECASE,
    )
    if bp_match:
        sbp = float(bp_match.group(1))
        dbp = float(bp_match.group(2))
        return round((sbp + 2.0 * dbp) / 3.0, 1)

    # Try direct MAP label
    map_match = re.search(
        r"map[:\s=]*(\d+\.?\d*)\s*(?:mmhg)?",
        text, re.IGNORECASE,
    )
    if map_match:
        return float(map_match.group(1))

    # Try bare numeric
    bare_match = re.search(r"\b(\d{2,3})\s*(?:mmhg)?\b", text, re.IGNORECASE)
    if bare_match:
        return float(bare_match.group(1))

    return None
