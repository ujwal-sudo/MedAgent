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
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


class DDxPlusLoader:
    """
    Loads and samples cases from the DDxPlus HuggingFace dataset using streaming.

    Uses uniform streaming sampling for fast, reliable access.
    """

    def __init__(self, split: str = "train") -> None:
        """
        Args:
            split: Dataset split to load ("train", "validation", "test").
        """
        logger.info("Loading DDxPlus dataset from HuggingFace (split=%s, streaming=True)...", split)
        self.dataset = load_dataset(
            "aai530-group6/ddxplus",
            split=split,
            streaming=True
        )
        self.iterator = iter(self.dataset)
        logger.info("Streaming dataset initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_case(self) -> dict[str, Any]:
        """
        Sample one case using uniform streaming.
        Cycles through the dataset when exhausted.
        """
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset iterator for cycling
            self.iterator = iter(self.dataset)
            return next(self.iterator)
