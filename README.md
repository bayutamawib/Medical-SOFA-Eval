# Medical SOFA Eval

**Calibrated Abstention Evaluation for Clinical LLMs**

A modular Python library for evaluating the clinical reasoning quality of Large Language Models (LLMs) through SOFA Score extraction verification, Cactus Signal routing compliance, and Calibrated Abstention scoring.

## Overview

`sofa-eval` is the evaluation backbone of the **Gemma-Sync** research project, which investigates how Reinforcement Learning from Verifiable Rewards (RLVR) and Group Relative Policy Optimization (GRPO) can instill *Calibrated Abstention* in Small Language Models (SLMs) for medical triage.

### What is Calibrated Abstention?

In medical AI, a wrong answer delivered with confidence is far more dangerous than no answer at all. **Calibrated Abstention** is the model's learned ability to:

1. **Recognize** when critical clinical data is missing or ambiguous.
2. **Signal** its uncertainty via the `<|escalate|>` routing token.
3. **Defer** the clinical decision to a human expert rather than hallucinate.

## Architecture

```
sofa-eval/
├── src/sofa_eval/
│   ├── __init__.py          # Public API
│   ├── utils.py             # Regex parsing, text extraction, SOFA table parser
│   ├── oracle.py            # SOFA score verification engine + abstention bonus
│   └── rewards.py           # 4-tier GRPO reward functions (RLVR)
│   └── SOFA_Syllabus_ENG.md # Syllabus for understanding SOFA (English)
│   └── SOFA_Syllabus_ID.md  # Syllabus for understanding SOFA (Bahasa Indonesia)
├── tests/
│   └── test_oracle.py       # Unit tests (pytest)
├── pyproject.toml           # Build configuration
└── README.md
```

## Reward Architecture

The library implements a 4-tier reward system for GRPO training:

| Reward Component         | Weight | Description                                      |
|--------------------------|--------|--------------------------------------------------|
| **Correctness (RLVR)**   | 0.50   | Binary exact-match of `\boxed{X}` vs ground truth |
| **SOFA Oracle**          | 0.20   | 6-system clinical verification engine             |
| **Format Compliance**    | 0.10   | LaTeX structure and step-marker validation         |
| **Process Quality**      | 0.20   | Chain-of-Thought depth and medical terminology     |
| **Abstention Bonus**     | +0.20  | Reward for appropriate `<\|escalate\|>` signaling  |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from sofa_eval import parse_sofa_table, score_sofa_oracle, reward_correctness

# Parse a model-generated SOFA table
text = """
| SOFA Component | Parameter  | Value     | Score |
|----------------|------------|-----------|-------|
| Respiratory    | PaO2/FiO2  | 350       | 1     |
| Coagulation    | Platelets  | 120       | 1     |
| Liver          | Bilirubin  | N/P       | N/P   |
| Cardiovascular | MAP        | 75 mmHg   | 0     |
| CNS            | GCS        | 15        | 0     |
| Renal          | Creatinine | 1.0 mg/dL | 0     |
"""

components = parse_sofa_table(text)
oracle_score = score_sofa_oracle(text)
print(f"Parsed {len(components)} components, Oracle score: {oracle_score:.3f}")
```

## Running Tests

```bash
pytest
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{bayutama2026sofaeval,
  title  = {sofa-eval: Calibrated Abstention Evaluation for Clinical LLMs},
  author = {Bayutama Wibisono, Narendra},
  year   = {2026},
  url    = {https://github.com/narendrabayutama/sofa-eval}
}
```

## References

- Vincent, J. L., et al. (1996). *The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure.* Intensive Care Medicine, 22(7), 707-710.
- DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948.
- Wibisono, N. B. (2026). Calibrated Abstention and Clinical Deferral in Small Language Models via RLVR and GRPO. Zenodo. https://doi.org/10.5281/zenodo.19913606

## License

MIT License. See [LICENSE](LICENSE) for details.
