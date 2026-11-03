from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DEFAULT_BCP_ROOT = "/path/to/MAS_Loop/datasets/BrowseComp-Plus"
_DEFAULT_BM25_INDEX = f"{_DEFAULT_BCP_ROOT}/scripts_build_index/indexes/bm25"
_GATE_CACHE_SCHEMA = "mas_loop.controller_parquet_bm25/1"
_GATE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_GATE_K1 = 1.2
_GATE_B = 0.75
_GATE_MAX_DOC_CHARS = 9000
_POSTING = struct.Struct("<II")


@dataclass
class EvidenceDoc:
    docid: str
    text: str
    score: float | None = None


def _bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _gate_tokens(text: str, max_chars: int | None = None) -> list[str]:
    text = text or ""
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return [m.group(0).lower() for m in _GATE_TOKEN_RE.finditer(text) if len(m.group(0)) > 1]


def _corpus_files(corpus_dir: Path) -> list[Path]:
    files = sorted(p for p in (corpus_dir / "data").glob("*.parquet") if not p.name.startswith("._"))
    if not files:
        files = sorted(p for p in corpus_dir.glob("*.parquet") if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No parquet corpus files found under {corpus_dir}")
    return files


def _corpus_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_cache_base(raw: str | None, bcp_root: Path) -> Path:
    default = bcp_root / ".mas_loop_cache" / "controller_parquet_bm25"
    requested = Path(raw).expanduser() if raw else default
    if not requested.is_absolute():
        requested = bcp_root / requested
    resolved_parent = requested.parent.resolve()
    resolved = resolved_parent / requested.name
    allowed_roots = [bcp_root.resolve(), Path("/tmp").resolve()]
    # Security: cache writes are restricted to the local BCP tree or /tmp so an
    # environment override cannot turn index creation into an arbitrary-file write.
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(
            "MAS_LOOP_BCP_BM25_CACHE_DIR must stay under the resolved BCP root or /tmp; "
            f"got {resolved}"
        )
    return resolved


class GateParquetBM25Retriever:
    """Exact retrieval controller pure-Python BM25 semantics with a safe persistent cache.

    retrieval controller persisted a Python object graph with pickle. This implementation
    deliberately uses Arrow IPC plus a fixed-width uint32 postings file, neither
    of which can instantiate Python objects while loading. Query-result caching
    remains disabled; only the immutable corpus index is persisted.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        *,
        cache_base: str | Path | None = None,
        k1: float = _GATE_K1,
        b: float = _GATE_B,
        max_doc_chars: int = _GATE_MAX_DOC_CHARS,
    ):
        self.corpus_dir = Path(corpus_dir).resolve()
        self.k1 = float(k1)
        self.b = float(b)
        self.max_doc_chars = int(max_doc_chars)
        if self.k1 <= 0 or not 0 <= self.b <= 1 or self.max_doc_chars <= 0:
            raise ValueError("invalid retrieval controller BM25 parameters")
        self.files = _corpus_files(self.corpus_dir)
        self.corpus_fingerprint = _corpus_fingerprint(self.files)
        base = _safe_cache_base(str(cache_base) if cache_base else None, self.corpus_dir.parent)
        config_key = hashlib.sha256(
            json.dumps(
                {
                    "schema": _GATE_CACHE_SCHEMA,
                    "corpus": self.corpus_fingerprint,
                    "k1": self.k1,
                    "b": self.b,
                    "max_doc_chars": self.max_doc_chars,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        self.cache_dir = base / config_key
        self.cache_hit = False
        self._load_or_build()

    def _metadata(self) -> dict[str, Any]:
        return {
            "schema_version": _GATE_CACHE_SCHEMA,
            "corpus_fingerprint": self.corpus_fingerprint,
            "k1": self.k1,
            "b": self.b,
            "max_doc_chars": self.max_doc_chars,
        }

    def _cache_complete(self) -> bool:
        expected = {"metadata.json", "documents.arrow", "terms.arrow", "postings.u32le"}
        if not self.cache_dir.is_dir() or self.cache_dir.is_symlink():
            return False
        return all((self.cache_dir / name).is_file() and not (self.cache_dir / name).is_symlink() for name in expected)

    def _load_or_build(self) -> None:
        base = self.cache_dir.parent
        base.mkdir(parents=True, exist_ok=True)
        lock_path = base / f".{self.cache_dir.name}.lock"
        # Build jobs are serial today, but flock also prevents accidental
        # concurrent launchers from publishing a partially written cache.
        import fcntl

        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self._cache_complete():
                self._load_cache()
                self.cache_hit = True
                return
            if self.cache_dir.exists():
                raise RuntimeError(f"incomplete BM25 cache exists; refusing to overwrite it: {self.cache_dir}")
            tmp_dir = base / f".{self.cache_dir.name}.tmp.{os.getpid()}"
            if tmp_dir.exists():
                raise RuntimeError(f"temporary BM25 cache path already exists: {tmp_dir}")
            tmp_dir.mkdir(mode=0o700)
            try:
                self._build_cache(tmp_dir)
                os.replace(tmp_dir, self.cache_dir)
            except BaseException:
                for name in ("metadata.json", "documents.arrow", "terms.arrow", "postings.u32le"):
                    try:
                        (tmp_dir / name).unlink()
                    except FileNotFoundError:
                        pass
                try:
                    tmp_dir.rmdir()
                except OSError:
                    pass
                raise
            self._load_cache()
            self.cache_hit = False

    def _build_cache(self, target: Path) -> None:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq

        docids: list[str] = []
        texts: list[str] = []
        doc_lens: list[int] = []
        df: dict[str, int] = defaultdict(int)
        inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for fp in self.files:
            table = pq.read_table(fp, columns=["docid", "text"])
            for row in table.to_pylist():
                docid = str(row.get("docid", ""))
                text = str(row.get("text", ""))
                did = len(docids)
                docids.append(docid)
                texts.append(text)
                counts = Counter(_gate_tokens(text, self.max_doc_chars))
                dl = sum(counts.values()) or 1
                doc_lens.append(dl)
                for term, tf in counts.items():
                    df[term] += 1
                    inverted[term].append((did, int(tf)))

        n = max(1, len(docids))
        avgdl = float(sum(doc_lens) / n) if doc_lens else 1.0
        idf = {term: math.log(1.0 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}
        docs_table = pa.table(
            {
                "docid": docids,
                "text": texts,
                "doc_len": pa.array(doc_lens, type=pa.uint32()),
            }
        )
        with (target / "documents.arrow").open("wb") as handle:
            with ipc.new_file(handle, docs_table.schema) as writer:
                writer.write_table(docs_table)

        terms: list[str] = []
        idfs: list[float] = []
        starts: list[int] = []
        counts: list[int] = []
        offset = 0
        with (target / "postings.u32le").open("wb", buffering=1024 * 1024) as handle:
            for term, rows in inverted.items():
                terms.append(term)
                idfs.append(idf[term])
                starts.append(offset)
                counts.append(len(rows))
                payload = bytearray(len(rows) * _POSTING.size)
                for row_index, (did, tf) in enumerate(rows):
                    _POSTING.pack_into(payload, row_index * _POSTING.size, did, tf)
                handle.write(payload)
                offset += len(rows)
        terms_table = pa.table(
            {
                "term": terms,
                "idf": pa.array(idfs, type=pa.float64()),
                "start": pa.array(starts, type=pa.uint64()),
                "count": pa.array(counts, type=pa.uint32()),
            }
        )
        with (target / "terms.arrow").open("wb") as handle:
            with ipc.new_file(handle, terms_table.schema) as writer:
                writer.write_table(terms_table)

        metadata = self._metadata() | {
            "num_docs": len(docids),
            "num_terms": len(terms),
            "num_postings": offset,
            "avgdl": avgdl,
            "corpus_files": [p.name for p in self.files],
        }
        (target / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def _load_cache(self) -> None:
        import mmap

        import pyarrow as pa
        import pyarrow.ipc as ipc

        # Security: only JSON, Arrow IPC, and fixed-width integers are decoded;
        # unlike Gate's pickle cache these formats cannot execute constructors.
        metadata = json.loads((self.cache_dir / "metadata.json").read_text(encoding="utf-8"))
        for key, expected in self._metadata().items():
            if metadata.get(key) != expected:
                raise RuntimeError(f"BM25 cache metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}")
        documents = ipc.open_file(pa.memory_map(str(self.cache_dir / "documents.arrow"), "r")).read_all()
        terms = ipc.open_file(pa.memory_map(str(self.cache_dir / "terms.arrow"), "r")).read_all()
        required_docs = {"docid", "text", "doc_len"}
        required_terms = {"term", "idf", "start", "count"}
        if set(documents.column_names) != required_docs or set(terms.column_names) != required_terms:
            raise RuntimeError("BM25 cache Arrow schema mismatch")
        self.docids = documents["docid"].to_pylist()
        self.texts = documents["text"].to_pylist()
        self.doc_lens = documents["doc_len"].to_pylist()
        self.avgdl = float(metadata["avgdl"])
        term_values = terms["term"].to_pylist()
        idf_values = terms["idf"].to_pylist()
        start_values = terms["start"].to_pylist()
        count_values = terms["count"].to_pylist()
        self.term_index = {
            term: (float(term_idf), int(start), int(count))
            for term, term_idf, start, count in zip(term_values, idf_values, start_values, count_values, strict=True)
        }
        postings_path = self.cache_dir / "postings.u32le"
        expected_size = int(metadata["num_postings"]) * _POSTING.size
        if postings_path.stat().st_size != expected_size:
            raise RuntimeError("BM25 postings cache size mismatch")
        self._postings_handle = postings_path.open("rb")
        self._postings_mmap = mmap.mmap(self._postings_handle.fileno(), 0, access=mmap.ACCESS_READ)

    def search(self, query: str, k: int) -> list[EvidenceDoc]:
        scores: dict[int, float] = defaultdict(float)
        terms = Counter(_gate_tokens(query))
        for term, qtf in terms.items():
            entry = self.term_index.get(term)
            if entry is None:
                continue
            idf, start, count = entry
            for position in range(start, start + count):
                did, tf = _POSTING.unpack_from(self._postings_mmap, position * _POSTING.size)
                dl = self.doc_lens[did]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[did] += qtf * idf * (tf * (self.k1 + 1.0) / denom)
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: int(k)]
        return [EvidenceDoc(docid=self.docids[did], text=self.texts[did], score=float(score)) for did, score in top]

    def status(self) -> dict[str, Any]:
        return {
            "backend": "controller_parquet",
            "cache_dir": str(self.cache_dir),
            "index_cache_hit": self.cache_hit,
            "corpus_fingerprint": self.corpus_fingerprint,
            "k1": self.k1,
            "b": self.b,
            "max_doc_chars": self.max_doc_chars,
            "tokenizer": r"[A-Za-z0-9]+; lowercase; token_length>1",
        }


# Backward-compatible name for callers/tests that explicitly imported the old fallback.
ParquetBM25Retriever = GateParquetBM25Retriever

_RETRIEVER: Any | None = None
_RETRIEVER_LOCK = threading.Lock()
_RETRIEVER_BACKEND = "uninitialized"
_RETRIEVER_INIT_SEC = 0.0
_RETRIEVER_DETAILS: dict[str, Any] = {}


def _setup_bcp_path(root: str) -> None:
    root_path = Path(root).resolve()
    for p in (root_path, root_path.parent.parent.parent if len(root_path.parents) >= 3 else None):
        if p is not None and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def get_bm25_retriever(index_path: str | None = None, bcp_root: str | None = None):
    """Return one process-resident retriever; never cache query results."""
    global _RETRIEVER, _RETRIEVER_BACKEND, _RETRIEVER_INIT_SEC, _RETRIEVER_DETAILS
    bcp_root = bcp_root or os.getenv("MAS_LOOP_BCP_ROOT", _DEFAULT_BCP_ROOT)
    index_path = index_path or os.getenv("MAS_LOOP_BM25_INDEX", _DEFAULT_BM25_INDEX)
    backend = os.getenv("MAS_LOOP_BCP_BM25_BACKEND", "auto").strip().lower()
    aliases = {"gate": "controller_parquet", "parquet": "controller_parquet", "pyserini": "lucene"}
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "lucene", "controller_parquet"}:
        raise ValueError(f"unsupported MAS_LOOP_BCP_BM25_BACKEND={backend!r}")
    with _RETRIEVER_LOCK:
        if _RETRIEVER is None:
            started = time.perf_counter()
            _setup_bcp_path(bcp_root)
            try:
                if backend == "controller_parquet":
                    corpus_dir = Path(bcp_root) / "browsecomp-plus-corpus"
                    _RETRIEVER = GateParquetBM25Retriever(
                        corpus_dir,
                        cache_base=os.getenv("MAS_LOOP_BCP_BM25_CACHE_DIR") or None,
                        k1=float(os.getenv("MAS_LOOP_BM25_K1", str(_GATE_K1))),
                        b=float(os.getenv("MAS_LOOP_BM25_B", str(_GATE_B))),
                        max_doc_chars=int(os.getenv("MAS_LOOP_BCP_BM25_MAX_DOC_CHARS", str(_GATE_MAX_DOC_CHARS))),
                    )
                    _RETRIEVER_BACKEND = "controller_parquet"
                    _RETRIEVER_DETAILS = _RETRIEVER.status()
                else:
                    try:
                        from retrieval import BM25Retriever  # type: ignore

                        _RETRIEVER = BM25Retriever(index_path)
                        _RETRIEVER_BACKEND = "lucene_pyserini"
                        _RETRIEVER_DETAILS = {"backend": _RETRIEVER_BACKEND, "index": str(index_path)}
                    except ModuleNotFoundError as exc:
                        allow = _bool_env("MAS_LOOP_ALLOW_PARQUET_BM25_FALLBACK", "1")
                        require = _bool_env("MAS_LOOP_REQUIRE_LUCENE_BM25", "0")
                        if backend == "lucene" or require or not allow:
                            raise RuntimeError(
                                "Lucene/Pyserini BM25 is required but unavailable; select "
                                "MAS_LOOP_BCP_BM25_BACKEND=controller_parquet to reproduce retrieval controller fallback"
                            ) from exc
                        corpus_dir = Path(bcp_root) / "browsecomp-plus-corpus"
                        _RETRIEVER = GateParquetBM25Retriever(
                            corpus_dir,
                            cache_base=os.getenv("MAS_LOOP_BCP_BM25_CACHE_DIR") or None,
                            k1=float(os.getenv("MAS_LOOP_BM25_K1", str(_GATE_K1))),
                            b=float(os.getenv("MAS_LOOP_BM25_B", str(_GATE_B))),
                            max_doc_chars=int(os.getenv("MAS_LOOP_BCP_BM25_MAX_DOC_CHARS", str(_GATE_MAX_DOC_CHARS))),
                        )
                        _RETRIEVER_BACKEND = "controller_parquet"
                        _RETRIEVER_DETAILS = _RETRIEVER.status()
            finally:
                _RETRIEVER_INIT_SEC = time.perf_counter() - started
        return _RETRIEVER


def bm25_backend_status(*, initialize: bool = False) -> dict[str, Any]:
    if initialize:
        get_bm25_retriever()
    return {
        "backend": _RETRIEVER_BACKEND,
        "initialization_sec": round(_RETRIEVER_INIT_SEC, 6),
        "initialized": _RETRIEVER is not None,
        "query_result_cache_enabled": False,
        "index_resident": _RETRIEVER is not None,
        **_RETRIEVER_DETAILS,
    }


def _reset_bm25_retriever_for_tests() -> None:
    global _RETRIEVER, _RETRIEVER_BACKEND, _RETRIEVER_INIT_SEC, _RETRIEVER_DETAILS
    with _RETRIEVER_LOCK:
        mm = getattr(_RETRIEVER, "_postings_mmap", None)
        handle = getattr(_RETRIEVER, "_postings_handle", None)
        if mm is not None:
            mm.close()
        if handle is not None:
            handle.close()
        _RETRIEVER = None
        _RETRIEVER_BACKEND = "uninitialized"
        _RETRIEVER_INIT_SEC = 0.0
        _RETRIEVER_DETAILS = {}


def _document_to_dict(document: Any) -> dict[str, Any]:
    if hasattr(document, "__dataclass_fields__"):
        item = asdict(document)
    else:
        item = {
            "docid": getattr(document, "docid", ""),
            "text": getattr(document, "text", ""),
            "score": getattr(document, "score", None),
        }
    return {
        "docid": str(item.get("docid", "")),
        "text": str(item.get("text", "")),
        "score": item.get("score"),
    }


def retrieve_bcp_evidence_with_meta(
    query: str, topk: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute every BM25 query afresh while keeping the corpus index resident."""
    topk = int(topk or os.getenv("MAS_LOOP_BM25_TOPK", "5"))
    query = str(query)
    if topk <= 0:
        return [], {
            "cache_hit": False,
            "cache_key_scope": "disabled",
            "query": query,
            "topk": topk,
        }
    retriever = get_bm25_retriever()
    rows = retriever.search(query, k=topk)
    docs = [_document_to_dict(row) for row in rows]
    return docs, {
        "cache_hit": False,
        "cache_key_scope": "disabled",
        "query_result_cache_enabled": False,
        "index_resident": True,
        "bm25_backend": _RETRIEVER_BACKEND,
        "bm25_backend_details": dict(_RETRIEVER_DETAILS),
        "retriever_initialization_sec": round(_RETRIEVER_INIT_SEC, 6),
        "query": query,
        "topk": topk,
        "index": os.getenv("MAS_LOOP_BM25_INDEX", _DEFAULT_BM25_INDEX),
    }


def retrieve_bcp_evidence(query: str, topk: int | None = None) -> list[dict[str, Any]]:
    documents, _ = retrieve_bcp_evidence_with_meta(query, topk)
    return documents


def format_evidence(docs: list[dict[str, Any]], max_chars_per_doc: int | None = None) -> str:
    max_chars_per_doc = int(max_chars_per_doc or os.getenv("MAS_LOOP_BM25_DOC_MAX_CHARS", "1200"))
    if not docs:
        return ""
    blocks = []
    for i, d in enumerate(docs, 1):
        text = (d.get("text") or "").replace("\n", " ").strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + " ..."
        score = d.get("score")
        score_s = f", score={score:.4f}" if isinstance(score, (int, float)) else ""
        blocks.append(f"[BM25-{i}] docid={d.get('docid','')}{score_s}\n{text}")
    return "\n\n".join(blocks)
