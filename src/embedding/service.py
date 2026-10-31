"""KN-CHUNK-02: embed chunks and queries with the active model through Amazon Bedrock.

Uses Knowledge's IAM role only (invariant 10). A failed call is retried with backoff, then fails
with a generic error; the Bedrock error and the text are never surfaced or logged.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from typing import Any, Protocol

from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from src.chunking.schemas import Chunk
from src.chunking.service import estimate_tokens
from src.config.knowledge_config import EmbeddingConfig
from src.core.exceptions import EmbeddingUnavailableError
from src.embedding.exceptions import EmbeddingInputTooLong, NonRetryableEmbeddingError
from src.embedding.schemas import EmbeddedChunk

logger = logging.getLogger(__name__)


class BedrockRuntime(Protocol):
    def invoke_model(self, **kwargs: Any) -> dict[str, Any]: ...


def chunk_input_text(chunk: Chunk) -> str:
    """The text sent to the model for a chunk. The heading path shapes the vector, so it is part of the input.

    The live evaluation (KN-CHUNK-05) records one vector per exact text, so this is the one place that builds it.
    """
    return f"{chunk.heading_path}\n\n{chunk.text}"


class Embedder:
    def __init__(
        self,
        client: BedrockRuntime,
        config: EmbeddingConfig = EmbeddingConfig(),
        *,
        sleep: Callable[[float], None] = time.sleep,
        count_tokens: Callable[[str], int] = estimate_tokens,
    ) -> None:
        self._client = client
        self._config = config
        self._sleep = sleep
        self._count_tokens = count_tokens

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        vector = self._embed(chunk_input_text(chunk))
        return EmbeddedChunk(chunk, vector, self._config.active_embedding_model)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """All or nothing: any failure raises, so the caller never holds a partial set."""
        return [self.embed_chunk(chunk) for chunk in chunks]

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Same call as chunks, without a heading prefix (KN-RET-03)."""
        return self._embed(query)

    def _embed(self, text: str) -> tuple[float, ...]:
        if self._count_tokens(text) > self._config.max_input_tokens:
            raise EmbeddingInputTooLong()
        attempts = len(self._config.retry_backoff_seconds) + 1
        for attempt in range(attempts):
            try:
                return self._invoke(text)
            except (NoCredentialsError, PartialCredentialsError) as exc:
                # No amount of waiting supplies a credential; fail now so the caller gets its 503 inside its budget.
                logger.error("Embedding call failed: %s", type(exc).__name__)
                raise EmbeddingUnavailableError(log_message="no AWS credentials for the embedding client") from None
            except NonRetryableEmbeddingError as exc:
                # A reason that waiting cannot change (a replay recording without this text): no backoff.
                logger.error("Embedding call failed: %s", type(exc).__name__)
                raise EmbeddingUnavailableError(log_message=str(exc)) from None
            except Exception as exc:  # noqa: BLE001 - any failure is retried, then reported generically
                logger.error("Embedding call failed (attempt %d of %d): %s", attempt + 1, attempts, type(exc).__name__)
                if attempt < attempts - 1:
                    self._sleep(self._config.retry_backoff_seconds[attempt])
        raise EmbeddingUnavailableError(log_message=f"embedding failed after {attempts} attempts")

    def _invoke(self, text: str) -> tuple[float, ...]:
        response = self._client.invoke_model(
            modelId=self._config.active_embedding_model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": self._config.embedding_dimensions,
                    "normalize": self._config.embedding_normalize,
                }
            ),
        )
        vector = [float(x) for x in json.loads(response["body"].read())["embedding"]]
        if len(vector) != self._config.embedding_dimensions:
            raise ValueError("unexpected embedding dimension count")
        norm = math.sqrt(sum(x * x for x in vector))
        if self._config.embedding_normalize and norm > 0 and not math.isclose(norm, 1.0, rel_tol=1e-6):
            vector = [x / norm for x in vector]
        return tuple(vector)
