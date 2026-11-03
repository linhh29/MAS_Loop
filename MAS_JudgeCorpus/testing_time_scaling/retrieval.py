from __future__ import annotations

import bisect
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any


class HotpotFtsRetriever:
    backend = "hotpotqa_lucene_export_fts5"
    def __init__(self, lucene_path: Path):
        self.lucene_path = lucene_path.resolve()
        self.sqlite_path = self.lucene_path.parent / "bm25_fts5" / "index.sqlite3"
        self.conn = sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True, check_same_thread=False)
    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        terms = [x for x in re.findall(r"[\w]+", query.lower()) if len(x) > 1][:32]
        match = " OR ".join(json.dumps(x) for x in terms or ["empty"])
        rows = self.conn.execute("SELECT docid,title,contents,bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?", (match, int(k))).fetchall()
        return [{"docid": str(d), "title": str(t or ""), "text": str(c or ""), "score": float(-s)} for d,t,c,s in rows]


class ParquetArrowRetriever:
    backend = "controller_parquet"
    def __init__(self, cache_root: Path, *, max_doc_chars: int = 9000):
        import numpy as np
        import pyarrow as pa
        import pyarrow.ipc as ipc
        roots = []
        for p in cache_root.glob("*/metadata.json"):
            meta = json.loads(p.read_text(encoding="utf-8"))
            if int(meta.get("max_doc_chars", 0)) == int(max_doc_chars): roots.append((p.parent, meta))
        if not roots: raise FileNotFoundError(f"no cache with max_doc_chars={max_doc_chars}: {cache_root}")
        self.root, self.meta = sorted(roots, key=lambda x: str(x[0]))[0]
        self.np = np
        self.doc_src, self.term_src = pa.memory_map(str(self.root / "documents.arrow"), "r"), pa.memory_map(str(self.root / "terms.arrow"), "r")
        self.docs = ipc.open_file(self.doc_src).read_all()
        self.docids, self.texts = self.docs.column("docid"), self.docs.column("text")
        total, self.ends = 0, []
        for chunk in self.texts.chunks: total += len(chunk); self.ends.append(total)
        terms = ipc.open_file(self.term_src).read_all().to_pydict()
        self.terms = {t: (float(i), int(s), int(c)) for t,i,s,c in zip(terms["term"], terms["idf"], terms["start"], terms["count"])}
        self.postings = np.memmap(self.root / "postings.u32le", dtype="<u4", mode="r").reshape(-1, 2)
        self.doc_len = self.docs.column("doc_len").combine_chunks().to_numpy(zero_copy_only=False)
        self.num_docs, self.avgdl = int(self.meta["num_docs"]), float(self.meta["avgdl"])
        self.k1, self.b = float(self.meta["k1"]), float(self.meta["b"])
    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        np = self.np
        terms = [(t, self.terms[t]) for t in dict.fromkeys(re.findall(r"[\w]+", query.lower())) if len(t)>1 and t in self.terms and self.terms[t][2] <= 60000]
        terms.sort(key=lambda x: x[1][0], reverse=True); terms = terms[:16]
        if not terms: return []
        scores = np.zeros(self.num_docs, dtype=np.float32)
        for _, (idf,start,count) in terms:
            pairs=self.postings[start:start+count]; idx=pairs[:,0].astype(np.int64,copy=False); tf=pairs[:,1].astype(np.float32,copy=False)
            norm=self.k1*(1-self.b+self.b*self.doc_len[idx]/self.avgdl); scores[idx]+=idf*(tf*(self.k1+1))/(tf+norm)
        positive=np.flatnonzero(scores>0)
        if not len(positive): return []
        n=min(max(1,int(k)),len(positive)); chosen=positive[np.argpartition(scores[positive],-n)[-n:]] if n<len(positive) else positive; chosen=chosen[np.argsort(scores[chosen])[::-1]]
        out=[]
        for raw in chosen:
            idx=int(raw); ci=bisect.bisect_right(self.ends,idx); start=0 if ci==0 else self.ends[ci-1]; li=idx-start
            out.append({"docid":str(self.docids.chunk(ci)[li].as_py()),"text":str(self.texts.chunk(ci)[li].as_py()),"score":float(scores[idx])})
        return out


def _configure_vendored_pyserini() -> None:
    """Prefer this experiment's isolated Pyserini and compact JDK runtime."""
    vendor = Path(__file__).resolve().parent / "vendor"
    if not vendor.is_dir():
        return
    vendor_str = str(vendor)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)

    java_home = vendor / "jdk4py" / "java-runtime"
    java_bin = java_home / "bin"
    if (java_bin / "java").is_file():
        # This must happen before pyjnius is imported and starts the JVM.
        os.environ["JAVA_HOME"] = str(java_home)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if str(java_bin) not in path_parts:
            os.environ["PATH"] = str(java_bin) + os.pathsep + os.environ.get("PATH", "")

    # jdk4py is intentionally jlink-minimized and omits the optional incubator
    # vector module. Sparse BM25 does not use it. The second flag prevents
    # optional dense encoders from importing the host Transformers package.
    os.environ.setdefault("PYSERINI_DISABLE_VECTOR_MODULE", "1")
    os.environ.setdefault("PYSERINI_SPARSE_ONLY", "1")


class HotpotLuceneRetriever:
    backend = "hotpotqa_lucene_pyserini"

    def __init__(self, lucene_path: Path, bcp_root: Path):
        del bcp_root  # Kept in the signature for config/backward compatibility.
        _configure_vendored_pyserini()
        from pyserini.search.lucene import LuceneSearcher

        self.lucene_path = lucene_path.resolve()
        self.searcher = LuceneSearcher(str(self.lucene_path))

    @classmethod
    def _extract_text(cls, raw: Any) -> str:
        """Unwrap both Lucene's stored JSON and the corpus's nested JSON."""
        value = raw
        for _ in range(4):
            if value is None:
                return ""
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            if isinstance(value, dict):
                if value.get("text") is not None:
                    return str(value["text"])
                if value.get("contents") is not None:
                    value = value["contents"]
                    continue
                if value.get("raw") is not None:
                    value = value["raw"]
                    continue
                return json.dumps(value, ensure_ascii=False)
            return str(value)
        return str(value)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for hit in self.searcher.search(query, k=int(k)):
            raw = None
            lucene_document = getattr(hit, "lucene_document", None)
            if lucene_document is not None:
                raw = lucene_document.get("contents") or lucene_document.get("raw")
            if raw is None:
                stored = self.searcher.doc(hit.docid)
                raw = stored.raw() if stored is not None else None
            out.append({"docid": str(hit.docid), "score": float(hit.score), "text": self._extract_text(raw)})
        return out


def build_retriever(dataset: str, mas_root: Path, *, allow_hotpot_fts_fallback: bool = False) -> Any:
    bcp = mas_root / "datasets/BrowseComp-Plus"
    if dataset == "hotpotqa":
        lucene = bcp / "hotpotqa/indexes/bm25"
        try:
            return HotpotLuceneRetriever(lucene, bcp)
        except Exception:
            if not allow_hotpot_fts_fallback: raise
            return HotpotFtsRetriever(lucene)
    if dataset == "bcp": return ParquetArrowRetriever(bcp / ".mas_loop_cache/controller_parquet_bm25", max_doc_chars=9000)
    raise ValueError(dataset)


def format_evidence(docs: list[dict[str, Any]], max_chars: int = 13500) -> str:
    blocks, used = [], 0
    for rank, doc in enumerate(docs, 1):
        text = " ".join(str(doc.get("text", "")).split())
        block = f"[rank={rank} docid={doc.get('docid','')} score={doc.get('score','')}] {text}"
        if used + len(block) > max_chars: block = block[:max(0, max_chars-used)]
        if block: blocks.append(block); used += len(block)
        if used >= max_chars: break
    return "\n\n".join(blocks)

async def multi_round_retrieve(*, dataset: str, question: str, initial_query: str, retriever: Any, client: Any | None, topk: int, rounds: int, max_chars: int, role: str, metadata: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Run the pinned retrieval profile: Hotpot legacy=1 round; BCP retrieval controller=5 rounds."""
    seen: set[str] = set()
    docs: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    query = initial_query.strip() or question
    for round_idx in range(max(1, int(rounds))):
        hits = retriever.search(query, int(topk))
        new_docs = []
        for doc in hits:
            docid = str(doc.get("docid", ""))
            if docid and docid not in seen:
                seen.add(docid); docs.append(doc); new_docs.append(doc)
        trajectory.append({"round": round_idx + 1, "query": query, "retrieved_docids": [str(x.get("docid", "")) for x in hits], "new_docids": [str(x.get("docid", "")) for x in new_docs], "documents": new_docs})
        if round_idx + 1 >= rounds or client is None:
            break
        preview = format_evidence(docs, min(max_chars, 6000))
        prompt = (
            "You are a BM25 retrieval gap detector. Do not answer the question. "
            "Given the question and evidence so far, output one concise lexical search query for the missing fact. "
            "Output only the query.\n\n"
            f"Question:\n{question}\n\nEvidence so far:\n{preview}"
        )
        query, _ = await client.generate(user_prompt=prompt, system_prompt="", role=role, temperature=0.0, max_tokens=128, metadata={**(metadata or {}), "retrieval_round": round_idx + 2})
        query = query.strip().splitlines()[0][:800] or question
    return format_evidence(docs, max_chars), trajectory
