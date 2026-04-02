"""
ddxplus_loader.py
-----------------
Loads the DDxPlus dataset from HuggingFace Hub using the `datasets` library.

Dataset: aai530-group6/ddxplus
Fields available: age, sex, pathology, evidences, initial_evidence, differential_diagnosis

Usage:
    loader = DDxPlusLoader()
    case = loader.sample_case()
"""

import logging
import random
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


class DDxPlusLoader:
    """
    Loads and samples cases from the DDxPlus HuggingFace dataset.

    Lazily keeps the dataset in memory after first load.
    All field access downstream is left to the consumer (PatientAgent).
    """

    def __init__(self, split: str = "train") -> None:
        """
        Args:
            split: Dataset split to load ("train", "validation", "test").
        """
        logger.info("Loading DDxPlus dataset from HuggingFace (split=%s)...", split)
        self.dataset = load_dataset("aai530-group6/ddxplus", split=split)
        logger.info("Loaded %d patient cases.", len(self.dataset))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of cases in the loaded split."""
        return len(self.dataset)

    def sample_case(self, rng: random.Random | None = None) -> dict[str, Any]:
        """
        Return one randomly sampled case dict.

        Args:
            rng: Optional seeded Random instance for reproducibility.
                 If None, uses the module-level random.

        Returns:
            Raw case dict from the HuggingFace dataset row.
        """
        if len(self.dataset) == 0:
            raise RuntimeError("Dataset is empty — cannot sample a case.")
        picker = rng or random
        idx = picker.randint(0, len(self.dataset) - 1)
        return dict(self.dataset[idx])

    def get_case(self, index: int) -> dict[str, Any]:
        """Return case at a specific index (for deterministic replay)."""
        if index < 0 or index >= len(self.dataset):
            raise IndexError(
                f"Index {index} out of range [0, {len(self.dataset) - 1}]."
            )
        return dict(self.dataset[index])
