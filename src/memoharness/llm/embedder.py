from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when the embedding API call fails in strict mode."""


class OpenAIEmbedder:
    """Wraps the OpenAI Embeddings API with an LRU cache.

    Args:
        client: An OpenAI client instance.
        model: Embedding model name.
        cache_maxsize: Maximum number of cached embeddings.
        strict: If ``True`` (default), API failures raise
            ``EmbeddingError`` so callers notice immediately.
            If ``False``, a zero-vector is returned and a warning is logged.
            The zero-vector is **not** cached, so the next call for the
            same text will retry the API.
    """

    def __init__(
        self,
        client,
        model: str = "text-embedding-3-small",
        cache_maxsize: int = 2048,
        strict: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.strict = strict
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_maxsize = cache_maxsize

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            vector = response.data[0].embedding
        except Exception as exc:
            if self.strict:
                raise EmbeddingError(
                    "Embedding API call failed for text (length={0}): {1}".format(
                        len(text), exc
                    )
                ) from exc
            logger.warning(
                "Embedding API call failed, returning zero-vector (not cached): %s",
                exc,
            )
            # Return zero-vector but do NOT cache it — next call will retry.
            return [0.0] * 1536

        self._cache[text] = vector
        if len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)

        return vector

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
