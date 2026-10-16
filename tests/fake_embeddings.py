"""Deterministic fake embedder for OFFLINE tests and demos.

Mechanical stand-in only: vectors are a hash-based bag-of-words, so texts
that share vocabulary get similar vectors. It proves wiring, ranking
order, filtering, deletion, and persistence — it says NOTHING about real
semantic quality (that is the job of the opt-in e2e tests against the
actual class endpoint on port 9002).
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Sequence


class FakeEmbedder:
    """Deterministic bag-of-words hash embedder.

    ``dim`` is generous (64) so vocabulary collisions are rare and ranking
    tests behave; the vectors are still meaningless semantically.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in text.lower().split():
                digest = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
                bucket = digest % self.dim
                vector[bucket] += 1.0 if (digest >> 8) % 2 else -1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append([x / norm for x in vector])
        return vectors

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
