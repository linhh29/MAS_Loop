from __future__ import annotations

import json
from pathlib import Path

from mas_loop_rl.executor import MASExecutor


def test_bm25_full_text_is_not_persisted_and_pair_metadata_is_bounded(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MAS_LOOP_RETRIEVAL_ARTIFACT_DIR", str(tmp_path / "must_not_exist"))
    monkeypatch.setenv("MAS_LOOP_BM25_DOC_PREVIEW_CHARS", "32")
    documents = [
        {"docid": "a", "score": 3.5, "text": "x" * 100_000},
        {"docid": "b", "score": 2.0, "text": "complete evidence"},
    ]

    compact, summary = MASExecutor._compact_retrieval(documents)

    assert all("text" not in row for row in compact)
    assert compact[0]["text_preview"] == "x" * 32
    assert compact[0]["text_length"] == 100_000
    assert len(json.dumps(compact)) < 1_000
    assert summary["persisted"] is False
    assert summary["reason"] == "bm25_query_result_cache_disabled"
    assert summary["document_count"] == 2
    assert not (tmp_path / "must_not_exist").exists()
