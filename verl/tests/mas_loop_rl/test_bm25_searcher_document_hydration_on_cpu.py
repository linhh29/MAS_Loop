from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


class _FakeLuceneSearcher:
    def __init__(self, _index_path):
        pass


def _load_bm25_searcher(monkeypatch):
    bcp_root = Path(__file__).resolve().parents[3] / "datasets" / "BrowseComp-Plus"
    monkeypatch.syspath_prepend(str(bcp_root))

    pyserini = types.ModuleType("pyserini")
    search = types.ModuleType("pyserini.search")
    lucene = types.ModuleType("pyserini.search.lucene")
    lucene.LuceneSearcher = _FakeLuceneSearcher
    monkeypatch.setitem(sys.modules, "pyserini", pyserini)
    monkeypatch.setitem(sys.modules, "pyserini.search", search)
    monkeypatch.setitem(sys.modules, "pyserini.search.lucene", lucene)

    sys.modules.pop("searcher.searchers.bm25_searcher", None)
    return importlib.import_module("searcher.searchers.bm25_searcher").BM25Searcher


class _StoredFields:
    def __init__(self, **fields):
        self.fields = fields

    def get(self, key):
        return self.fields.get(key)


class _StoredDocument:
    def __init__(self, raw):
        self._raw = raw

    def raw(self):
        return self._raw


class _Hit:
    def __init__(self, docid, fields, score=1.0):
        self.docid = docid
        self.score = score
        self.lucene_document = _StoredFields(**fields)


class _Backend:
    def __init__(self, hits, documents):
        self.hits = hits
        self.documents = documents
        self.doc_calls = []

    def search(self, query, k):
        return self.hits[:k]

    def doc(self, docid):
        self.doc_calls.append(docid)
        return self.documents.get(docid)


def _instance(cls, backend):
    obj = object.__new__(cls)
    obj.searcher = backend
    return obj


def test_search_extracts_direct_contents_without_document_fallback(monkeypatch):
    cls = _load_bm25_searcher(monkeypatch)
    backend = _Backend([_Hit("d1", {"contents": "plain evidence"})], {})

    rows = _instance(cls, backend).search("query", k=1)

    assert rows[0]["text"] == "plain evidence"
    assert backend.doc_calls == []


def test_search_extracts_contents_nested_in_raw_json(monkeypatch):
    cls = _load_bm25_searcher(monkeypatch)
    raw = '{"id":"d1","contents":"hydrated corpus evidence"}'
    backend = _Backend([_Hit("d1", {"raw": raw})], {})

    rows = _instance(cls, backend).search("query", k=1)

    assert rows[0]["text"] == "hydrated corpus evidence"
    assert backend.doc_calls == []


def test_search_falls_back_to_stored_document_raw(monkeypatch):
    cls = _load_bm25_searcher(monkeypatch)
    backend = _Backend(
        [_Hit("d1", {})],
        {"d1": _StoredDocument('{"contents":"stored fallback evidence"}')},
    )

    rows = _instance(cls, backend).search("query", k=1)

    assert rows[0]["text"] == "stored fallback evidence"
    assert backend.doc_calls == ["d1"]


def test_missing_document_returns_empty_text_without_stringifying_none(monkeypatch):
    cls = _load_bm25_searcher(monkeypatch)
    backend = _Backend([_Hit("missing", {})], {})

    rows = _instance(cls, backend).search("query", k=1)

    assert rows[0]["text"] == ""
    assert rows[0]["text"] != "None"


def test_get_document_supports_text_and_contents_json_fields(monkeypatch):
    cls = _load_bm25_searcher(monkeypatch)
    backend = _Backend([], {
        "text": _StoredDocument('{"text":"text field"}'),
        "contents": _StoredDocument('{"contents":"contents field"}'),
    })
    searcher = _instance(cls, backend)

    assert searcher.get_document("text") == {"docid": "text", "text": "text field"}
    assert searcher.get_document("contents") == {"docid": "contents", "text": "contents field"}
    assert searcher.get_document("absent") is None
