from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _stable_task_key(row: dict[str, Any], line_number: int) -> str:
    for field in ("task_id", "game_file"):
        value = str(row.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    raise ValueError(
        f"ALFWorld row {line_number} has neither a non-empty task_id nor game_file"
    )


def stable_unique_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"ALFWorld row {line_number} must be a JSON object")
            key = _stable_task_key(row, line_number)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def write_unique_manifest(input_path: Path, output_path: Path, expected_rows: int) -> int:
    rows = stable_unique_rows(input_path)
    if expected_rows > 0 and len(rows) != expected_rows:
        raise ValueError(
            f"expected exactly {expected_rows} unique ALFWorld tasks, found {len(rows)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stable-deduplicate an ALFWorld JSONL manifest by task_id/game_file."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-rows", type=int, default=4032)
    args = parser.parse_args()
    if args.expected_rows < 0:
        raise SystemExit("--expected-rows must be non-negative")

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        raise SystemExit("input and output paths must differ")
    if not input_path.is_file():
        raise SystemExit(f"missing ALFWorld input manifest: {input_path}")

    try:
        count = write_unique_manifest(input_path, output_path, args.expected_rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"alfworld unique manifest: input={input_path} output={output_path} rows={count}")


if __name__ == "__main__":
    main()
