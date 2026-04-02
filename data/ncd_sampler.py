"""
ncd_sampler.py
--------------
NCDSampler — NCD-weighted sampling over DDxPlusLoader.

Buckets cases by pathology into 5 NCD categories and samples
according to configurable weights for curriculum-aware training.
"""

import random
from typing import Any

# NCD sampling weights — must sum to 1.0
NCD_WEIGHTS = {
    "diabetes": 0.30,
    "hypertension": 0.25,
    "comorbid": 0.20,
    "other_ncd": 0.10,
    "general": 0.15,
}

assert abs(sum(NCD_WEIGHTS.values()) - 1.0) < 1e-9, "NCD_WEIGHTS must sum to 1.0"


def _categorize_pathology(pathology: str) -> str:
    """Bucket a pathology string into one of 5 NCD categories."""
    p = pathology.lower()
    has_diabet = "diabet" in p
    has_hypertens = "hypertens" in p

    if has_diabet and has_hypertens:
        return "comorbid"
    if has_diabet:
        return "diabetes"
    if has_hypertens:
        return "hypertension"

    # Other cardiovascular / renal NCDs
    ncd_keywords = ["cardio", "heart", "renal", "kidney", "stroke", "coronary", "atrial"]
    for kw in ncd_keywords:
        if kw in p:
            return "other_ncd"

    return "general"


class NCDSampler:
    """
    Wraps a loader that exposes `sample_case(rng)` and `size`.
    Buckets the full dataset by NCD category and samples
    according to NCD_WEIGHTS.
    """

    def __init__(self, loader: Any, seed: int | None = None) -> None:
        self._loader = loader
        self._rng = random.Random(seed)
        self._buckets: dict[str, list[int]] = {k: [] for k in NCD_WEIGHTS}
        self._build_buckets()

    def _build_buckets(self) -> None:
        """Index every case into its NCD bucket at init time."""
        for idx in range(self._loader.size):
            case = self._loader.get_case(idx)
            pathology = str(
                case.get("PATHOLOGY", case.get("pathology", ""))
            )
            cat = _categorize_pathology(pathology)
            self._buckets[cat].append(idx)

    def sample(self) -> dict[str, Any]:
        """
        Sample one case using NCD-weighted category selection.
        1. Pick a category according to NCD_WEIGHTS.
        2. Uniformly sample a case index within that category.
        Falls back to uniform random if the chosen bucket is empty.
        """
        categories = list(NCD_WEIGHTS.keys())
        weights = [NCD_WEIGHTS[c] for c in categories]
        chosen_cat = self._rng.choices(categories, weights=weights, k=1)[0]

        bucket = self._buckets[chosen_cat]
        if not bucket:
            # Fallback: sample uniformly from entire dataset
            idx = self._rng.randint(0, self._loader.size - 1)
        else:
            idx = self._rng.choice(bucket)

        return self._loader.get_case(idx)

    def reseed(self, seed: int) -> None:
        """Re-seed for reproducible episodes (called by MedEnv.reset(seed=N))."""
        self._rng.seed(seed)

    @property
    def dataset_size(self) -> int:
        return self._loader.size
