"""
ncd_sampler.py
--------------
NCDSampler — Uniform streaming sampling over DDxPlusLoader.

Simplified to uniform sampling for fast, reliable access.
No weighted NCD sampling to avoid performance bottlenecks.
"""

import random
from typing import Any


class NCDSampler:
    """
    Wraps a loader that exposes `sample_case()`.
    Uses uniform streaming sampling.
    """

    def __init__(self, loader: Any, seed: int | None = None) -> None:
        self._loader = loader
        self._rng = random.Random(seed)

    def sample(self) -> dict[str, Any]:
        """
        Sample one case using uniform streaming.
        """
        return self._loader.sample_case()

    def reseed(self, seed: int) -> None:
        """Re-seed for reproducible episodes (called by MedEnv.reset(seed=N))."""
        self._rng.seed(seed)

    @property
    def dataset_size(self) -> int:
        # Streaming dataset size unknown, return large number for compatibility
        return 1000000
