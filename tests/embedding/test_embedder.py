"""KN-CHUNK-02: embedding model and call.

__author__ = "Van Vo"
"""

from __future__ import annotations

import io
import json
import math
from typing import Any

import pytest

from src.chunking.schemas import Chunk
from src.config.knowledge_config import EmbeddingConfig
from src.core.exceptions import EmbeddingUnavailableError
from src.embedding.exceptions import EmbeddingInputTooLong
from src.embedding.service import Embedder

DIMS = 1024
SECRET_TEXT = "Account 4111-1111-1111-1111 balance"


def unit(position: int = 0, dims: int = DIMS) -> list[float]:
    vector = [0.0] * dims
    vector[position] = 1.0
    return vector


class FakeBedrock:
    """Stands in for the boto3 bedrock-runtime client; each call pops the next outcome."""

    def __init__(self, *outcomes: list[float] | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return {"body": io.BytesIO(json.dumps({"embedding": outcome, "inputTextTokenCount": 5}).encode())}


def make_chunk(text: str = "Confirm the columns.", heading: str = "Statements > CSV mapping") -> Chunk:
    return Chunk(source_path="help/a.md", heading_path=heading, text=text, ordinal=0, token_count=4)


def embedder(client: FakeBedrock, config: EmbeddingConfig | None = None) -> tuple[Embedder, list[float]]:
    sleeps: list[float] = []
    return Embedder(client, config or EmbeddingConfig(), sleep=sleeps.append), sleeps


def test_chunk_gets_a_normalized_vector_tagged_with_the_active_model() -> None:
    emb, _ = embedder(FakeBedrock(unit(3)))
    result = emb.embed_chunk(make_chunk())
    assert len(result.embedding) == DIMS
    assert math.isclose(math.sqrt(sum(x * x for x in result.embedding)), 1.0, rel_tol=1e-6)
    assert result.embedding_model == "amazon.titan-embed-text-v2:0"
    assert result.chunk == make_chunk()


def test_embedded_text_is_heading_path_blank_line_chunk_text() -> None:
    client = FakeBedrock(unit())
    embedder(client)[0].embed_chunk(make_chunk())
    body = json.loads(client.calls[0]["body"])
    assert body["inputText"] == "Statements > CSV mapping\n\nConfirm the columns."
    assert body["dimensions"] == DIMS and body["normalize"] is True


def test_model_id_comes_from_config_not_from_code() -> None:
    client = FakeBedrock(unit(dims=512))
    config = EmbeddingConfig(active_embedding_model="other.model:1", embedding_dimensions=512)
    result = embedder(client, config)[0].embed_chunk(make_chunk())
    assert client.calls[0]["modelId"] == "other.model:1"
    assert result.embedding_model == "other.model:1" and len(result.embedding) == 512


def test_query_is_embedded_without_a_heading_prefix() -> None:
    client = FakeBedrock(unit())
    vector = embedder(client)[0].embed_query("How do I confirm a CSV mapping?")
    assert json.loads(client.calls[0]["body"])["inputText"] == "How do I confirm a CSV mapping?"
    assert len(vector) == DIMS


def test_vector_that_is_not_unit_length_is_normalized() -> None:
    vector = [0.0] * DIMS
    vector[0] = 3.0
    vector[1] = 4.0
    result = embedder(FakeBedrock(vector))[0].embed_chunk(make_chunk())
    assert math.isclose(result.embedding[0], 0.6) and math.isclose(result.embedding[1], 0.8)


def test_text_over_the_model_limit_is_rejected_before_any_call() -> None:
    client = FakeBedrock(unit())
    config = EmbeddingConfig(max_input_tokens=10)
    emb = Embedder(client, config, sleep=lambda _s: None, count_tokens=lambda t: len(t.split()))
    with pytest.raises(EmbeddingInputTooLong):
        emb.embed_chunk(make_chunk(text="word " * 50))
    assert client.calls == []


def test_failed_call_is_retried_three_times_with_backoff_then_succeeds() -> None:
    client = FakeBedrock(RuntimeError("throttled"), RuntimeError("throttled"), RuntimeError("throttled"), unit())
    emb, sleeps = embedder(client)
    assert len(emb.embed_chunk(make_chunk()).embedding) == DIMS
    assert len(client.calls) == 4 and sleeps == [1, 2, 4]


def test_exhausted_retries_fail_with_embedding_unavailable_and_no_internal_detail() -> None:
    client = FakeBedrock(RuntimeError(f"arn:aws:bedrock secret {SECRET_TEXT}"))
    emb, sleeps = embedder(client)
    with pytest.raises(EmbeddingUnavailableError) as raised:
        emb.embed_chunk(make_chunk())
    assert raised.value.code == "embedding_unavailable"
    assert len(client.calls) == 4 and sleeps == [1, 2, 4]
    assert "arn:aws" not in str(raised.value) and SECRET_TEXT not in str(raised.value)


def test_response_with_wrong_dimension_count_is_treated_as_a_failure() -> None:
    client = FakeBedrock(unit(dims=512))
    emb, _ = embedder(client)
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_chunk(make_chunk())


def test_embedding_a_batch_is_all_or_nothing() -> None:
    client = FakeBedrock(unit(), RuntimeError("down"))
    emb, _ = embedder(client)
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_chunks([make_chunk("one"), make_chunk("two")])


def test_embedding_a_batch_returns_one_result_per_chunk_in_order() -> None:
    emb, _ = embedder(FakeBedrock(unit(1)))
    chunks = [make_chunk("one"), make_chunk("two")]
    assert [r.chunk for r in emb.embed_chunks(chunks)] == chunks


def test_failure_logs_never_contain_chunk_text(caplog: pytest.LogCaptureFixture) -> None:
    emb, _ = embedder(FakeBedrock(RuntimeError("boom")))
    with caplog.at_level("DEBUG"), pytest.raises(EmbeddingUnavailableError):
        emb.embed_chunk(make_chunk(text=SECRET_TEXT))
    assert SECRET_TEXT not in caplog.text


def test_missing_credentials_fail_at_once_without_retries() -> None:
    """KN-FAPI-05: retrying a credential that is not there only delays the 503."""
    from botocore.exceptions import NoCredentialsError

    client = FakeBedrock(NoCredentialsError())
    emb, sleeps = embedder(client)
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_query("How do I confirm a CSV mapping?")
    assert len(client.calls) == 1 and sleeps == []
