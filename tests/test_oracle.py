"""
tests/test_oracle.py — Unit Tests for SOFA Oracle Verification Engine
======================================================================

Tests cover:
1. Valid, well-formatted Markdown SOFA tables
2. Tables with messy whitespace and inconsistent formatting
3. Tables containing N/P (Not Provided) abstention markers
4. Tables with vasopressor keywords triggering CV scoring
"""

import pytest

from sofa_eval.utils import parse_sofa_table, extract_boxed, check_sofa_not_applicable
from sofa_eval.oracle import (
    score_sofa_oracle,
    score_abstention_bonus,
    validate_cv_score,
    SOFA_COMPONENTS,
)
from sofa_eval.rewards import reward_correctness, reward_format


class TestParseValidTable:
    """Test parsing of a well-formatted Markdown SOFA table."""

    VALID_TABLE = """\
| SOFA Component   | Parameter       | Value        | Score |
|------------------|-----------------|--------------|-------|
| Respiratory      | PaO2/FiO2      | 350          | 1     |
| Coagulation      | Platelets       | 120          | 1     |
| Liver            | Bilirubin       | 1.5 mg/dL    | 1     |
| Cardiovascular   | MAP             | 65 mmHg      | 1     |
| CNS              | GCS             | 14           | 1     |
| Renal            | Creatinine      | 1.0 mg/dL    | 0     |
"""

    def test_parse_returns_all_six_components(self):
        result = parse_sofa_table(self.VALID_TABLE)
        assert len(result) == 6
        for comp in SOFA_COMPONENTS:
            assert comp in result, f"Missing component: {comp}"

    def test_parse_extracts_correct_values(self):
        result = parse_sofa_table(self.VALID_TABLE)
        assert "350" in result["Respiratory"]["value"]
        assert result["Renal"]["score"].strip() == "0"

    def test_oracle_scores_valid_table_above_zero(self):
        score = score_sofa_oracle(self.VALID_TABLE)
        assert score > 0.5, f"Expected score > 0.5, got {score}"


class TestMessyWhitespace:
    """Test parsing with irregular spacing and padding."""

    MESSY_TABLE = """\
|   Respiratory    |   PaO2/FiO2    |    450       |   0   |
|  Coagulation     | Platelets      |  80          | 2     |
| Liver            |Bilirubin       | 3.0 mg/dL    |2      |
|Cardiovascular    | MAP            |  75 mmHg     | 0     |
|   CNS            | GCS            | 15           | 0     |
|Renal             | Creatinine     | 0.8 mg/dL    | 0     |
"""

    def test_parse_handles_messy_spacing(self):
        result = parse_sofa_table(self.MESSY_TABLE)
        assert len(result) == 6, f"Expected 6 components, got {len(result)}"
        assert result["Coagulation"]["score"].strip() == "2"

    def test_oracle_handles_messy_table(self):
        score = score_sofa_oracle(self.MESSY_TABLE)
        assert score > 0.0, f"Expected positive score, got {score}"


class TestNPAbstention:
    """Test tables with N/P (Not Provided) markers for abstention."""

    NP_TABLE = """\
| SOFA Component   | Parameter       | Value        | Score |
|------------------|-----------------|--------------|-------|
| Respiratory      | PaO2/FiO2      | N/P          | N/P   |
| Coagulation      | Platelets       | N/P          | N/P   |
| Liver            | Bilirubin       | N/P          | N/P   |
| Cardiovascular   | MAP             | 120/80       | 0     |
| CNS              | GCS             | 15           | 0     |
| Renal            | Creatinine      | 1.0 mg/dL    | 0     |

<|escalate|>
"""

    def test_np_values_parsed_correctly(self):
        result = parse_sofa_table(self.NP_TABLE)
        assert result["Respiratory"]["value"].strip() == "N/P"
        assert result["Respiratory"]["score"].strip() == "N/P"

    def test_abstention_bonus_with_np_and_escalate(self):
        bonus = score_abstention_bonus(self.NP_TABLE)
        assert bonus > 0.0, f"Expected positive bonus, got {bonus}"

    def test_oracle_accepts_np_as_valid_score(self):
        score = score_sofa_oracle(self.NP_TABLE)
        assert score > 0.0


class TestVasopressorDetection:
    """Test cardiovascular scoring with vasopressor keywords."""

    VASOPRESSOR_TABLE = """\
| SOFA Component   | Parameter       | Value                         | Score |
|------------------|-----------------|-------------------------------|-------|
| Respiratory      | PaO2/FiO2      | 200                           | 2     |
| Coagulation      | Platelets       | 90                            | 2     |
| Liver            | Bilirubin       | 3.5 mg/dL                     | 2     |
| Cardiovascular   | MAP             | 55 mmHg, on norepinephrine    | 3     |
| CNS              | GCS             | 10                            | 2     |
| Renal            | Creatinine      | 2.5 mg/dL                     | 2     |
"""

    def test_vasopressor_detected_in_cv_value(self):
        result = parse_sofa_table(self.VASOPRESSOR_TABLE)
        cv_value = result["Cardiovascular"]["value"]
        assert "norepinephrine" in cv_value.lower()

    def test_cv_score_validated_with_vasopressor(self):
        is_plausible = validate_cv_score(
            "55 mmHg, on norepinephrine", claimed_score=3,
        )
        assert is_plausible is True

    def test_cv_score_rejects_high_map_with_high_score(self):
        is_plausible = validate_cv_score(
            "MAP: 85 mmHg", claimed_score=3,
        )
        assert is_plausible is False


class TestExtractBoxed:
    """Test LaTeX boxed answer extraction."""

    def test_standard_boxed(self):
        assert extract_boxed(r"The answer is \boxed{A}") == "A"

    def test_dollar_wrapped(self):
        assert extract_boxed(r"$$\boxed{C}$$") == "C"

    def test_lowercase_normalized(self):
        assert extract_boxed(r"\boxed{b}") == "B"

    def test_no_boxed_returns_none(self):
        assert extract_boxed("The answer is A.") is None


class TestRewardCorrectness:
    """Test the correctness reward function."""

    def test_correct_answer(self):
        rewards = reward_correctness(
            prompts=["Q1"],
            completions=[r"The answer is \boxed{A}"],
            answer=["A"],
        )
        assert rewards == [1.0]

    def test_incorrect_answer(self):
        rewards = reward_correctness(
            prompts=["Q1"],
            completions=[r"The answer is \boxed{B}"],
            answer=["A"],
        )
        assert rewards == [0.0]


class TestSofaNotApplicable:
    """Test SOFA_NOT_APPLICABLE detection."""

    def test_detects_flag(self):
        assert check_sofa_not_applicable("SOFA_NOT_APPLICABLE") is True

    def test_detects_natural_language(self):
        assert check_sofa_not_applicable("SOFA is not applicable here") is True

    def test_no_flag(self):
        assert check_sofa_not_applicable("The SOFA score is 5.") is False
