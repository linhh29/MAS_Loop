from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mas_loop_rl.bcp_retrieval as bcp

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tok(text: str, max_chars: int | None = None) -> list[str]:
    if max_chars and len(text or "") > max_chars:
        text = (text or "")[:max_chars]
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "") if len(m.group(0)) > 1]


def _write_corpus(root: Path, rows: list[dict[str, str]]) -> Path:
    corpus = root / "browsecomp-plus-corpus"
    data = corpus / "data"
    data.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), data / "train-00000-of-00001.parquet")
    return corpus


def _gate_reference(rows: list[dict[str, str]], query: str, k: int) -> list[tuple[str, float]]:
    k1, b, max_doc_chars = 1.2, 0.75, 9000
    doc_lens: list[int] = []
    inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
    df: dict[str, int] = defaultdict(int)
    for did, row in enumerate(rows):
        counts = Counter(_tok(row["text"], max_doc_chars))
        doc_lens.append(sum(counts.values()) or 1)
        for term, tf in counts.items():
            df[term] += 1
            inverted[term].append((did, tf))
    n = max(1, len(rows))
    avgdl = sum(doc_lens) / n
    idf = {term: math.log(1.0 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}
    scores: dict[int, float] = defaultdict(float)
    for term, qtf in Counter(_tok(query)).items():
        for did, tf in inverted.get(term, []):
            denom = tf + k1 * (1.0 - b + b * doc_lens[did] / avgdl)
            scores[did] += qtf * idf[term] * (tf * (k1 + 1.0) / denom)
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return [(rows[did]["docid"], score) for did, score in top]


def test_gate_backend_matches_reference_and_reloads_safe_cache(tmp_path: Path):
    rows = [
        {"docid": "d0", "text": "A x alpha alpha beta"},
        {"docid": "d1", "text": "alpha beta beta gamma"},
        {"docid": "d2", "text": "gamma delta and alpha"},
    ]
    corpus = _write_corpus(tmp_path, rows)
    cache = tmp_path / "cache"
    expected = _gate_reference(rows, "x alpha beta beta", 3)

    first = bcp.GateParquetBM25Retriever(corpus, cache_base=cache)
    first_rows = first.search("x alpha beta beta", 3)
    assert [row.docid for row in first_rows] == [docid for docid, _ in expected]
    assert [row.score for row in first_rows] == pytest.approx([score for _, score in expected], rel=0, abs=1e-15)
    assert first.cache_hit is False
    assert not list(cache.rglob("*.pkl"))

    second = bcp.GateParquetBM25Retriever(corpus, cache_base=cache)
    second_rows = second.search("x alpha beta beta", 3)
    assert [row.docid for row in second_rows] == [row.docid for row in first_rows]
    assert [row.score for row in second_rows] == pytest.approx([row.score for row in first_rows], rel=0, abs=0)
    assert second.cache_hit is True
    assert second.status()["k1"] == 1.2
    assert second.status()["b"] == 0.75
    assert second.status()["max_doc_chars"] == 9000


def test_cache_override_cannot_write_outside_bcp_or_tmp(tmp_path: Path):
    corpus = _write_corpus(tmp_path, [{"docid": "d0", "text": "alpha beta"}])
    with pytest.raises(ValueError, match="must stay under"):
        bcp.GateParquetBM25Retriever(corpus, cache_base="/var/lib/mas-loop-cache")


def test_explicit_backend_selection_records_gate_fingerprint(tmp_path: Path, monkeypatch):
    _write_corpus(tmp_path, [{"docid": "d0", "text": "alpha beta"}])
    monkeypatch.setenv("MAS_LOOP_BCP_ROOT", str(tmp_path))
    monkeypatch.setenv("MAS_LOOP_BCP_BM25_BACKEND", "controller_parquet")
    monkeypatch.setenv("MAS_LOOP_BCP_BM25_CACHE_DIR", str(tmp_path / "cache"))
    bcp._reset_bm25_retriever_for_tests()
    try:
        retriever = bcp.get_bm25_retriever()
        assert retriever.search("alpha", 1)[0].docid == "d0"
        status = bcp.bm25_backend_status()
        assert status["backend"] == "controller_parquet"
        assert status["k1"] == 1.2
        assert status["b"] == 0.75
        assert status["max_doc_chars"] == 9000
        assert status["query_result_cache_enabled"] is False
    finally:
        bcp._reset_bm25_retriever_for_tests()
