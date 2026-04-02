"""
generate_sample_dataset.py
--------------------------
Creates a minimal sample DDxPlus-compatible JSON dataset for local testing.

Run this ONCE before running main.py if you don't have the real dataset:
    python data/generate_sample_dataset.py

Outputs: data/sample_cases.json
"""

import json
import random
from pathlib import Path

SAMPLE_PATHOLOGIES = [
    "Pneumonia",
    "Viral pharyngitis",
    "Bronchitis",
    "Urinary tract infection",
    "Panic attack",
    "Pulmonary embolism",
    "Anemia",
    "GERD",
]

SAMPLE_EVIDENCES = [
    "E_1", "E_2", "E_5", "E_10", "E_12", "E_18", "E_23", "E_31", "E_42",
]

SEX_OPTIONS = ["M", "F"]


def generate_case(rng: random.Random) -> dict:
    pathology = rng.choice(SAMPLE_PATHOLOGIES)
    num_evidences = rng.randint(3, 8)
    evidences = rng.sample(SAMPLE_EVIDENCES, min(num_evidences, len(SAMPLE_EVIDENCES)))
    initial_evidence = rng.choice(evidences)

    return {
        "AGE": rng.randint(18, 80),
        "SEX": rng.choice(SEX_OPTIONS),
        "PATHOLOGY": pathology,
        "EVIDENCES": evidences,
        "INITIAL_EVIDENCE": initial_evidence,
        "DIFFERENTIAL_DIAGNOSIS": [
            [pathology, round(rng.uniform(0.5, 0.95), 3)],
            [rng.choice(SAMPLE_PATHOLOGIES), round(rng.uniform(0.05, 0.4), 3)],
        ],
    }


def main() -> None:
    rng = random.Random(42)
    num_cases = 50
    cases = [generate_case(rng) for _ in range(num_cases)]

    output_path = Path(__file__).parent / "sample_cases.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"✅ Generated {num_cases} sample cases → {output_path.resolve()}")


if __name__ == "__main__":
    main()
