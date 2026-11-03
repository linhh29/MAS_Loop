from __future__ import annotations
import argparse, hashlib, json, os, random, shutil, tempfile
from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow.parquet as pq
try:
    from .prompts import orchestra_prompt, editor_prompt
except ImportError:
    from mas_loop_rl.prompts import orchestra_prompt, editor_prompt

def _parquet_value_kind(obj: Any) -> str:
    """Return Arrow-relevant kinds without treating bool as an integer."""
    if isinstance(obj, dict):
        return "dict"
    if isinstance(obj, list):
        return "list"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, str):
        return "str"
    if isinstance(obj, bytes):
        return "bytes"
    return type(obj).__name__


def _collect_parquet_kinds(obj: Any, path: tuple[str, ...], kinds: dict[tuple[str, ...], set[str]]) -> None:
    if obj is None:
        return
    kinds.setdefault(path, set()).add(_parquet_value_kind(obj))
    if isinstance(obj, dict):
        for key, value in obj.items():
            _collect_parquet_kinds(value, path + (str(key),), kinds)
    elif isinstance(obj, list):
        for value in obj:
            _collect_parquet_kinds(value, path + ("[]",), kinds)


def _json_scalar(value: Any) -> str:
    """Deterministically stringify a value selected by batch schema analysis."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def _normalize_parquet_value(
    obj: Any,
    path: tuple[str, ...],
    stringify_paths: set[tuple[str, ...]],
) -> Any:
    if obj is None:
        return None
    if path in stringify_paths:
        return _json_scalar(obj)
    if isinstance(obj, dict):
        return {
            key: _normalize_parquet_value(value, path + (str(key),), stringify_paths)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [
            _normalize_parquet_value(value, path + ("[]",), stringify_paths)
            for value in obj
        ]
    return obj


def _sanitize_rows_for_parquet(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize incompatible nested leaf types across one Parquet batch.

    PyArrow assigns one type to every nested struct/list path. Real model traces
    occasionally report the same metadata leaf as an integer in one trajectory
    and a placeholder string in another. Normalizing each row independently
    cannot resolve that cross-row conflict. This batch pass preserves dict/list
    structure and only stringifies paths whose observed kinds are incompatible;
    integers mixed with floats remain numeric.
    """
    kinds: dict[tuple[str, ...], set[str]] = {}
    for row in rows:
        _collect_parquet_kinds(row, (), kinds)

    stringify_paths: set[tuple[str, ...]] = set()
    for path, observed in kinds.items():
        if len(observed) <= 1 or observed <= {"int", "float"}:
            continue
        stringify_paths.add(path)

    normalized = [
        _normalize_parquet_value(row, (), stringify_paths)
        for row in rows
    ]
    return [_sanitize_for_parquet(row) for row in normalized]


def _sanitize_for_parquet(obj: Any) -> Any:
    """Replace empty structs after batch-level type normalization."""
    if isinstance(obj, dict):
        out = {k: _sanitize_for_parquet(v) for k, v in obj.items()}
        return out or {"_": ""}
    if isinstance(obj, list):
        return [_sanitize_for_parquet(x) for x in obj]
    return obj


def read_records(path: Path):
    """Read either JSONL or a JSON list/dict dataset.

    BCP uses jsonl while DeepScaleR math training uses a single JSON list.
    Keeping both in one reader avoids format-specific launcher hacks.
    """
    suffix = path.suffix.lower()
    if suffix == '.json':
        with path.open('r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for x in obj:
                if isinstance(x, dict):
                    yield x
        elif isinstance(obj, dict):
            # Common wrappers: {data:[...]}, {train:[...]}, etc.
            for key in ('data', 'train', 'examples', 'items'):
                val = obj.get(key)
                if isinstance(val, list):
                    for x in val:
                        if isinstance(x, dict):
                            yield x
                    return
            yield obj
        return
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception:
                    pass

def default_mas():
    return {"agents":[{"id":"solver","role":"Solve the task step by step and produce final answer.","tools":[],"budget":1},{"id":"verifier","role":"Verify the solver answer and output corrected final answer.","tools":[],"budget":1}],"edges":[{"from":"solver","to":"verifier","type":"answer_check"}],"aggregation":"verifier","topology":"sequential","max_cost":10.0,"max_iterations":3}

def row_for(role: str, task: str, answer: str, idx: int, src_mas=None, src_traj=None, constraints=None, extra_meta=None):
    constraints = constraints or {"_": ""}
    prompt = orchestra_prompt(task, constraints) if role == 'orchestra' else editor_prompt(task, src_mas or default_mas(), src_traj or {"final_output":"","success":False,"agent_traces":[]}, constraints)
    extra_info={"index":idx,"role":role,"task_text":task,"constraints":constraints,"src_mas":src_mas or default_mas(),"src_trajectory":src_traj or {"_": ""}}
    if extra_meta:
        extra_info.update(extra_meta)
    return {"data_source":"mas_loop_"+role,"prompt":prompt,"reward_model":{"style":"rule","ground_truth":answer},"extra_info":extra_info}

def load_tasks(path: Path, role: str, dataset: str = "auto"):
    rows=[]
    for i,o in enumerate(read_records(path)):
        if 'task' in o and ('src_mas' in o or 'updated_mas' in o):
            task = o.get('task',{}).get('task_text') or o.get('task',{}).get('prompt') or json.dumps(o.get('task',{}), ensure_ascii=False)
            ans = str(o.get('task',{}).get('expected_answer') or o.get('expected_answer') or '')
            if role == 'editor':
                rows.append(row_for(role, task, ans, i, o.get('src_mas') or default_mas(), o.get('src_trajectory') or {}, o.get('edit_constraints') or {}))
            else:
                rows.append(row_for(role, task, ans, i, constraints=o.get('edit_constraints') or {}))
        else:
            # BrowseComp-Plus rows use {query_id, query, answer, gold_docs, ...}.
            # Training prompts/rewards must not leak gold_docs; they are kept only
            # as compact metadata for traceability. Real evidence is retrieved
            # online by BM25 from the local corpus during reward execution.
            is_alfworld = dataset == 'alfworld' or (dataset == 'auto' and str(o.get('dataset', '')).lower() == 'alfworld')
            is_hotpotqa = dataset == 'hotpotqa' or (dataset == 'auto' and ('supporting_facts' in o or 'hotpotqa' in str(path).lower()))
            if is_alfworld:
                descriptions = o.get('task_descriptions') or []
                task = str(descriptions[0] if descriptions else o.get('task_text') or o.get('question') or '').strip()
                if not task:
                    raise ValueError(f'ALFWorld row {i} has no task description')
                game_file = str(o.get('game_file') or '').strip()
                if not game_file:
                    raise ValueError(f'ALFWorld row {i} has no game_file')
                constraints = {
                    'dataset': 'alfworld',
                    'interactive_environment': True,
                    'requires_retrieval': False,
                    'answer_format': 'Success is determined only by the live environment.',
                }
                extra_meta = {
                    key: o.get(key)
                    for key in (
                        'dataset', 'id', 'task_id', 'game_file', 'env_step_limit',
                        'task_type', 'official_split', 'split', 'task_descriptions',
                        'floor_plan', 'traj_file',
                    )
                    if key in o
                }
                extra_meta['dataset'] = 'alfworld'
                rows.append(row_for(role, task, '', i, constraints=constraints, extra_meta=extra_meta))
            elif is_hotpotqa and 'answer' in o and ('query' in o or 'question' in o):
                task = str(o.get('query') or o.get('question') or '')
                ans = str(o.get('answer') or '')
                constraints = {
                    'dataset': 'hotpotqa',
                    'requires_retrieval': True,
                    'retriever': 'hotpotqa_bm25',
                    'answer_format': 'Return the final answer concisely after reasoning.'
                }
                extra_meta = {
                    'query_id': o.get('query_id') or o.get('id'),
                    'dataset': 'hotpotqa',
                    'type': o.get('type'),
                    'level': o.get('level'),
                    # Trace metadata only; never interpolate supporting facts into prompts.
                    'supporting_facts': o.get('supporting_facts') or [],
                }
                rows.append(row_for(role, task, ans, i, constraints=constraints, extra_meta=extra_meta))
            elif 'query' in o and 'answer' in o:
                task = str(o.get('query') or '')
                ans = str(o.get('answer') or '')
                constraints = {
                    'dataset': 'browsecomp_plus',
                    'requires_retrieval': True,
                    'retriever': 'bm25',
                    'answer_format': 'Return the final answer concisely after reasoning.'
                }
                extra_meta = {
                    'query_id': o.get('query_id'),
                    'dataset': 'browsecomp_plus',
                    'gold_docids': [str(d.get('docid','')) for d in (o.get('gold_docs') or []) if isinstance(d, dict)],
                    'evidence_docids': [str(d.get('docid','')) for d in (o.get('evidence_docs') or []) if isinstance(d, dict)],
                }
                rows.append(row_for(role, task, ans, i, constraints=constraints, extra_meta=extra_meta))
            elif 'problem' in o and 'answer' in o:
                # DeepScaleR math rows use {problem, answer, solution}. Do not
                # leak the reference solution into the prompt; keep only answer
                # for rule reward and compact metadata for traceability.
                task = str(o.get('problem') or '')
                ans = str(o.get('answer') or '')
                constraints = {
                    'dataset': 'deepscaler',
                    'domain': 'math',
                    'requires_retrieval': False,
                    'answer_format': 'Solve step by step and put the final answer in \\boxed{}.'
                }
                extra_meta = {
                    'dataset': 'deepscaler',
                    'has_reference_solution': bool(o.get('solution')),
                }
                rows.append(row_for(role, task, ans, i, constraints=constraints, extra_meta=extra_meta))
            else:
                task = o.get('task_text') or o.get('question') or o.get('prompt') or o.get('problem') or json.dumps(o, ensure_ascii=False)[:2000]
                ans = str(o.get('expected_answer') or o.get('answer') or o.get('target') or '')
                rows.append(row_for(role, task, ans, i))
    return rows

def _validate_published_parquet(path: Path, expected_rows: int) -> dict[str, Any]:
    """Fail before training if shared-storage publication produced an empty/corrupt artifact."""
    if not path.exists() or path.stat().st_size <= 0:
        raise IOError(f"published parquet is missing or empty: {path}")
    try:
        # pandas/pyarrow returns a zero-column DataFrame with length 0 for
        # ``columns=[]`` even when the parquet metadata contains rows. Read
        # the authoritative parquet metadata instead.
        actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise IOError(f"published parquet cannot be read: {path}: {exc}") from exc
    if expected_rows > 0 and actual_rows == 0:
        raise IOError(f"published parquet has zero rows: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "rows": actual_rows, "bytes": path.stat().st_size, "sha256": digest}


def _write_parquet_via_local_staging(frame: pd.DataFrame, destination: Path) -> None:
    """Write with pyarrow on local disk, then atomically publish to shared storage.

    The shared filesystem accepts normal copies but pyarrow's direct
    output stream can return EIO. Staging in /tmp avoids that filesystem API
    incompatibility while the final artifact remains under the requested work
    directory.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(os.environ.get("MAS_LOOP_PARQUET_STAGING_DIR", "/tmp/mas_loop_parquet_stage"))
    staging_dir.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(prefix=f"{destination.stem}.", suffix=".parquet", dir=staging_dir)
    os.close(fd)
    staged = Path(staged_name)
    publishing = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        frame.to_parquet(staged)
        shutil.copyfile(staged, publishing)
        os.replace(publishing, destination)
    finally:
        staged.unlink(missing_ok=True)
        publishing.unlink(missing_ok=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='JSONL tasks or MAS-EvoTraj JSONL')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--role', choices=['orchestra','editor','both'], default='both')
    ap.add_argument('--dataset', choices=['auto','bcp','deepscaler','hotpotqa','alfworld'], default='auto', help='Explicit dataset profile; avoids treating HotpotQA query rows as BrowseComp-Plus.')
    ap.add_argument('--val-ratio', type=float, default=0.05, help='Validation split ratio. Use 0 to keep every input row in train; val parquet will reuse train[:1].')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument(
        '--shuffle-train',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Shuffle loaded rows before the validation split. Use --no-shuffle-train for a stable source-order epoch.',
    )
    ap.add_argument('--max-rows', type=int, default=0, help='Optional smoke-test cap after loading and before shuffling. 0 means all rows.')
    ap.add_argument('--pad-train-to-multiple', type=int, default=0, help='If >0, repeat shuffled train rows so every original row is consumed by drop_last trainers.')
    args=ap.parse_args(); random.seed(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    roles=['orchestra','editor'] if args.role=='both' else [args.role]
    for role in roles:
        rows=load_tasks(Path(args.input), role, dataset=args.dataset)
        if args.max_rows and args.max_rows > 0:
            rows = rows[:args.max_rows]
        if args.shuffle_train:
            random.shuffle(rows)
        
        if args.val_ratio <= 0:
            nval = 0
        else:
            nval = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 1 else 0
        val = rows[:nval]
        train = rows[nval:] or rows
        original_train_len = len(train)
        if args.pad_train_to_multiple and args.pad_train_to_multiple > 0 and train:
            rem = len(train) % args.pad_train_to_multiple
            if rem:
                need = args.pad_train_to_multiple - rem
                for j in range(need):
                    dup = dict(train[j % len(train)])
                    extra = dict(dup.get('extra_info') or {})
                    extra['is_padding_duplicate'] = True
                    extra['padding_duplicate_of'] = extra.get('index')
                    extra['padding_duplicate_seq'] = j
                    extra['original_train_len'] = original_train_len
                    dup['extra_info'] = extra
                    train.append(dup)
        _write_parquet_via_local_staging(
            pd.DataFrame(_sanitize_rows_for_parquet(train)), out / f'{role}_train.parquet'
        )
        val_path = out / f'{role}_val.parquet'
        _write_parquet_via_local_staging(
            pd.DataFrame(_sanitize_rows_for_parquet(val or train[:1])), val_path
        )
        manifest = {
            "dataset": args.dataset, "role": role,
            "train": _validate_published_parquet(out / f'{role}_train.parquet', len(train)),
            "val": _validate_published_parquet(val_path, len(val or train[:1])),
        }
        (out / f'{role}_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        pad_note = f' (original_train={original_train_len}, padded_to_multiple={args.pad_train_to_multiple})' if args.pad_train_to_multiple else ''
        print(f'{role}: train={len(train)} val={len(val or train[:1])}{pad_note} -> {out}')
if __name__=='__main__': main()
