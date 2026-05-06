#!/usr/bin/env python3
"""
demo.py — sofa-eval Library Demo
=================================

Simulates two realistic LLM-generated clinical responses and runs them
through the full sofa-eval verification pipeline:
  1. SOFA table parsing
  2. Oracle scoring (5-axis deterministic verifier)
  3. Calibrated Abstention bonus detection

Run:  python demo.py
"""

from sofa_eval import (
    parse_sofa_table,
    score_sofa_oracle,
    score_abstention_bonus,
    extract_boxed,
    SOFA_COMPONENTS,
)

# ═══════════════════════════════════════════════════════════════════════════
# Simulated LLM Outputs
# ═══════════════════════════════════════════════════════════════════════════

SCENARIO_A = """\
## Step 1: Clinical Data Extraction

The patient is a 63-year-old male admitted to the ICU with community-acquired
pneumonia progressing to septic shock. Arterial Blood Gas (ABG) shows
PaO2 of 72 mmHg on FiO2 0.50 (P/F ratio = 144). Labs reveal platelets
at 88 ×10³/µL, bilirubin 1.8 mg/dL, and creatinine 2.4 mg/dL. GCS is 13
(E4 V4 M5). Hemodynamics: MAP 62 mmHg, currently on norepinephrine
0.15 µg/kg/min.

## Step 2: SOFA Assessment

| SOFA Component   | Parameter    | Extracted Value                        | SOFA Score |
|------------------|------------- |----------------------------------------|------------|
| Respiratory      | PaO2/FiO2   | 144 (PaO2 72, FiO2 0.50)              | 3          |
| Coagulation      | Platelets    | 88 ×10³/µL                             | 2          |
| Liver            | Bilirubin    | 1.8 mg/dL                              | 1          |
| Cardiovascular   | MAP/Vasop.   | MAP 62, norepinephrine 0.15 µg/kg/min  | 3          |
| CNS              | GCS          | 13                                     | 1          |
| Renal            | Creatinine   | 2.4 mg/dL                              | 2          |

**Total SOFA Score:** 12/24

## Step 3: Clinical Reasoning

The P/F ratio of 144 is consistent with moderate ARDS (Berlin criteria: 100-200).
The combination of respiratory failure, vasopressor-dependent shock, and
multi-organ involvement indicates severe sepsis. The coagulopathy (platelets 88)
suggests early DIC. GCS of 13 shows mild encephalopathy, likely
sepsis-associated.

**Confidence:** High confidence — all 6 parameters directly available.

<|local_ok|>

## Step 4: Final Answer

Given the clinical presentation of septic shock with ARDS and multi-organ
dysfunction, the most likely diagnosis is **severe sepsis with ARDS secondary
to community-acquired pneumonia**.

\\boxed{B}
"""

SCENARIO_B = """\
## Step 1: Clinical Data Extraction

A 45-year-old female presents post-cholecystectomy with persistent fever
and RUQ pain. Available labs: bilirubin 6.8 mg/dL (direct 4.2), platelets
42 ×10³/µL. ABG and ventilator settings were not documented in the chart.
Creatinine and GCS not recorded in the provided vignette.

## Step 2: SOFA Assessment

| SOFA Component   | Parameter    | Extracted Value         | SOFA Score |
|------------------|------------- |-------------------------|------------|
| Respiratory      | PaO2/FiO2   | N/P                     | N/P        |
| Coagulation      | Platelets    | 42 ×10³/µL              | 3          |
| Liver            | Bilirubin    | 6.8 mg/dL               | 3          |
| Cardiovascular   | MAP/Vasop.   | [assumed: stable]       | 0          |
| CNS              | GCS          | N/P                     | N/P        |
| Renal            | Creatinine   | N/P                     | N/P        |

**Total SOFA Score:** 6/24

**Reason:** 3 of 6 SOFA parameters (Respiratory, CNS, Renal) are not
provided in the clinical vignette. Scoring is incomplete; escalation
is recommended.

**Confidence:** Low confidence — significant missing data.

<|escalate|>

## Step 3: Clinical Reasoning

The elevated bilirubin with direct predominance suggests obstructive
pathophysiology, possibly a retained common bile duct stone or post-operative
biliary leak. The thrombocytopenia (42 ×10³/µL) raises concern for
sepsis-induced DIC. However, without respiratory, neurological, and renal
data, a complete SOFA assessment is not possible.

## Step 4: Final Answer

\\boxed{C}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Display Helpers
# ═══════════════════════════════════════════════════════════════════════════

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"


def color_score(value: float, max_val: float = 1.0) -> str:
    """Return a color-coded score string."""
    ratio = value / max_val if max_val else 0
    if ratio >= 0.8:
        color = GREEN
    elif ratio >= 0.5:
        color = YELLOW
    else:
        color = RED
    return f"{color}{BOLD}{value:.4f}{RESET}"


def print_header(title: str) -> None:
    width = 64
    print(f"\n{CYAN}{'═' * width}{RESET}")
    print(f"{CYAN}║{RESET}  {BOLD}{title}{RESET}")
    print(f"{CYAN}{'═' * width}{RESET}")


def print_section(label: str) -> None:
    print(f"\n  {MAGENTA}▸ {label}{RESET}")
    print(f"  {DIM}{'─' * 58}{RESET}")


def print_kv(key: str, value: str, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{DIM}{key:<24}{RESET} {value}")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(name: str, llm_output: str) -> None:
    """Run the full sofa-eval pipeline on a simulated LLM output."""

    print_header(name)

    # --- 1. Parse SOFA Table ---
    print_section("Parsed SOFA Table")

    components = parse_sofa_table(llm_output)

    if components:
        # Table header
        print(f"    {'Component':<18} {'Parameter':<14} {'Value':<28} {'Score':<6}")
        print(f"    {'─' * 18} {'─' * 14} {'─' * 28} {'─' * 6}")

        for comp_name in SOFA_COMPONENTS:
            if comp_name in components:
                row = components[comp_name]
                val_display = row['value'] if len(row['value']) <= 26 else row['value'][:23] + "..."
                score_str = row['score']

                # Color N/P scores in yellow, numeric in green/red
                if score_str.upper() in ("N/P", "NOT PROVIDED"):
                    score_colored = f"{YELLOW}{score_str}{RESET}"
                else:
                    score_colored = f"{GREEN}{score_str}{RESET}"

                print(f"    {comp_name:<18} {row['parameter']:<14} {val_display:<28} {score_colored}")
            else:
                print(f"    {comp_name:<18} {DIM}— not found —{RESET}")

        print(f"\n    {DIM}Components found: {len(components)}/{len(SOFA_COMPONENTS)}{RESET}")
    else:
        print(f"    {RED}No SOFA table detected in output.{RESET}")

    # --- 2. Oracle Score ---
    print_section("Oracle Score  (deterministic clinical verifier)")

    oracle_score = score_sofa_oracle(llm_output)
    print_kv("Oracle Score:", color_score(oracle_score))
    print_kv("Breakdown:", f"{DIM}table(0.25) + coverage(0.20) + validity(0.20)"
                           f" + plausibility(0.20) + arithmetic(0.15){RESET}")

    # --- 3. Abstention Bonus ---
    print_section("Calibrated Abstention Bonus")

    abstention_bonus = score_abstention_bonus(llm_output)

    # Detect signals for display
    np_count = llm_output.count("N/P")
    has_escalate = "<|escalate|>" in llm_output
    has_local_ok = "<|local_ok|>" in llm_output
    has_reason   = "**Reason:**" in llm_output

    print_kv("N/P markers found:", f"{BOLD}{np_count}{RESET}")
    print_kv("Escalate token:", f"{GREEN}yes{RESET}" if has_escalate else f"{DIM}no{RESET}")
    print_kv("Local-OK token:", f"{GREEN}yes{RESET}" if has_local_ok else f"{DIM}no{RESET}")
    print_kv("Reason provided:", f"{GREEN}yes{RESET}" if has_reason else f"{DIM}no{RESET}")
    print_kv("Abstention Bonus:", color_score(abstention_bonus, max_val=0.20) if abstention_bonus > 0
                                  else f"{DIM}0.0000  (insufficient signals){RESET}")

    # --- 4. Combined Score ---
    print_section("Combined Result")

    combined = min(oracle_score + abstention_bonus, 1.2)
    boxed = extract_boxed(llm_output)

    print_kv("Oracle + Abstention:", f"{BOLD}{combined:.4f}{RESET}  /  1.2000")
    print_kv("Extracted Answer:", f"{BOLD}{boxed}{RESET}" if boxed else f"{RED}not found{RESET}")

    # Quality verdict
    if combined >= 0.90:
        verdict = f"{GREEN}{BOLD}■ EXCELLENT{RESET}"
    elif combined >= 0.70:
        verdict = f"{GREEN}■ GOOD{RESET}"
    elif combined >= 0.50:
        verdict = f"{YELLOW}■ ACCEPTABLE{RESET}"
    else:
        verdict = f"{RED}■ POOR — needs improvement{RESET}"
    print_kv("Verdict:", verdict)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{BOLD}{'─' * 64}{RESET}")
    print(f"{BOLD}  sofa-eval  v0.1.0  —  Clinical LLM Evaluation Demo{RESET}")
    print(f"{BOLD}{'─' * 64}{RESET}")

    run_pipeline(
        "Scenario A — Complete SOFA (septic shock, high confidence)",
        SCENARIO_A,
    )

    run_pipeline(
        "Scenario B — Partial Data + Calibrated Abstention",
        SCENARIO_B,
    )

    print(f"\n{DIM}{'─' * 64}{RESET}")
    print(f"{DIM}  Demo complete. See src/sofa_eval/ for library source.{RESET}")
    print(f"{DIM}{'─' * 64}{RESET}\n")


if __name__ == "__main__":
    main()
