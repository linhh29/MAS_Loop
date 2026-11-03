from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .accounting import sum_usage
from .io_utils import atomic_json

DISPLAY = {
    "mas_lb": "MAS (LB)",
    "mas_majority_voting": "MAS Majority Voting",
    "mas_oracle_ub": "MAS Oracle (UB)",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "deepseek_v4_flash_trajectory": "DeepSeek-V4-Flash",
    "kimi_k3": "Kimi-K3",
    "kimi_k3_trajectory": "Kimi-K3",
    "qwen3_8_2_4t_a95b": "Qwen3.8-2.4T-A95B",
    "qwen3_8_2_4t_a95b_trajectory": "Qwen3.8-2.4T-A95B",
    "skywork_v2_qwen3_8b": "Skywork-V2-Qwen3-8B",
    "orch_rm": "Orch-RM",
    "mas_judgerm_orchestration": "MAS-JudgeRM (Ours)",
    "mas_judgerm_trajectory": "MAS-JudgeRM (Ours)",
}

EXPECTED_SAMPLES = {"hotpotqa": 200, "bcp": 168}

PAPER_ROWS = [
    ("Baseline", "mas_lb", ""),
    ("Baseline", "mas_majority_voting", ""),
    ("Baseline", "mas_oracle_ub", ""),
    ("LLM-as-a-Judge", "deepseek_v4_flash", "Orchestration"),
    ("LLM-as-a-Judge", "deepseek_v4_flash_trajectory", "Trajectory"),
    ("LLM-as-a-Judge", "kimi_k3", "Orchestration"),
    ("LLM-as-a-Judge", "kimi_k3_trajectory", "Trajectory"),
    ("LLM-as-a-Judge", "qwen3_8_2_4t_a95b", "Orchestration"),
    ("LLM-as-a-Judge", "qwen3_8_2_4t_a95b_trajectory", "Trajectory"),
    ("Reward Model", "skywork_v2_qwen3_8b", "Orchestration"),
    ("Reward Model", "orch_rm", "Orchestration"),
    ("Reward Model", "mas_judgerm_orchestration", "Orchestration"),
    ("Reward Model", "mas_judgerm_trajectory", "Trajectory"),
]


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Main-table token cost measures selector/judge inference only. Candidate-bank
    # generation is shared across selectors and retained separately for audit.
    selector_usage = sum_usage([x.get("selector_usage") for x in rows])
    end_to_end_usage = sum_usage([x.get("usage") for x in rows])
    scored = [x for x in rows if x.get("correct") is not None]
    return {
        "num_samples": len(rows),
        "num_scored": len(scored),
        "accuracy": sum(bool(x["correct"]) for x in scored) / len(scored) if scored else None,
        **selector_usage,
        "token_m": round(selector_usage["total_tokens"] / 1_000_000, 2),
        "token_scope": "selector_only",
        "end_to_end_input_tokens": end_to_end_usage["input_tokens"],
        "end_to_end_output_tokens": end_to_end_usage["output_tokens"],
        "end_to_end_total_tokens": end_to_end_usage["total_tokens"],
        "end_to_end_token_m": round(end_to_end_usage["total_tokens"] / 1_000_000, 2),
        "end_to_end_num_calls": end_to_end_usage["num_calls"],
        "end_to_end_usage_sources": end_to_end_usage["usage_sources"],
    }


def _latex_escape(text: str) -> str:
    return str(text).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _fmt_acc(value: Any) -> str:
    return "--" if value is None else f"{100 * float(value):.2f}\\%"


def _fmt_tokens(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _write_paper_tables(run_dir: Path, long_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(x["method_id"], x["dataset"]): x for x in long_table}
    active_methods = {x["method_id"] for x in long_table}
    rows: list[dict[str, Any]] = []
    for category, method_id, verification_level in PAPER_ROWS:
        if method_id not in active_methods:
            continue
        hp = lookup.get((method_id, "hotpotqa"))
        bcp = lookup.get((method_id, "bcp"))
        rows.append({
            "category": category,
            "method": DISPLAY[method_id],
            "method_id": method_id,
            "verification_level": verification_level,
            "token_scope": "selector_only",
            "hotpotqa_accuracy": None if hp is None else hp["accuracy"],
            "hotpotqa_token_m": None if hp is None else hp["token_m"],
            "bcp_oot_accuracy": None if bcp is None else bcp["accuracy"],
            "bcp_oot_token_m": None if bcp is None else bcp["token_m"],
        })
    atomic_json(run_dir / "paper_table.json", rows)
    cols = [
        "category", "method", "method_id", "verification_level", "token_scope",
        "hotpotqa_accuracy", "hotpotqa_token_m", "bcp_oot_accuracy", "bcp_oot_token_m",
    ]
    with (run_dir / "paper_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows({
            **row,
            "hotpotqa_token_m": _fmt_tokens(row["hotpotqa_token_m"]),
            "bcp_oot_token_m": _fmt_tokens(row["bcp_oot_token_m"]),
        } for row in rows)

    category_counts = {category: sum(1 for item in rows if item["category"] == category) for category in {item["category"] for item in rows}}
    method_counts = {(row["category"], row["method"]): sum(1 for item in rows if item["category"] == row["category"] and item["method"] == row["method"]) for row in rows}
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\multirow{2.5}{*}{Category} & \multirow{2.5}{*}{Method} & \multirow{2.5}{*}{Verification Level} & \multicolumn{2}{c}{HotpotQA} & \multicolumn{2}{c}{BCP\textsubscript{OOT}} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r" & & & Acc & \#Selector Token (M) & Acc & \#Selector Token (M) \\",
        r"\midrule",
    ]
    previous_category = None
    previous_method_key = None
    for row in rows:
        new_category = row["category"] != previous_category
        category = rf"\multirow{{{category_counts[row['category']]}}}{{*}}{{{_latex_escape(row['category'])}}}" if new_category else ""
        if previous_category is not None and row["category"] != previous_category:
            lines.append(r"\midrule")
        method_key = (row["category"], row["method"])
        method = "" if method_key == previous_method_key else _latex_escape(row["method"])
        if method and row["method_id"].startswith("mas_judgerm_"):
            method = r"\textbf{" + method + "}"
        if method and method_counts[method_key] > 1:
            method = rf"\multirow{{{method_counts[method_key]}}}{{*}}{{{method}}}"
        lines.append(
            f"{category} & {method} & {_latex_escape(row['verification_level'])} & "
            f"{_fmt_acc(row['hotpotqa_accuracy'])} & {_fmt_tokens(row['hotpotqa_token_m'])} & "
            f"{_fmt_acc(row['bcp_oot_accuracy'])} & {_fmt_tokens(row['bcp_oot_token_m'])} " + r"\\"
        )
        previous_category = row["category"]
        previous_method_key = method_key
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (run_dir / "paper_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def write_final_tables(run_dir: Path, method_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    complete_methods = {
        method for method, by_dataset in method_rows.items()
        if all(len(by_dataset.get(dataset, [])) == expected for dataset, expected in EXPECTED_SAMPLES.items())
    }
    table: list[dict[str, Any]] = []
    for method, by_dataset in method_rows.items():
        if method not in complete_methods:
            continue
        for dataset, rows in by_dataset.items():
            summary = summarize_method(rows)
            atomic_json(run_dir / f"methods/{method}/{dataset}/summary.json", summary)
            table.append({"method": DISPLAY.get(method, method), "method_id": method, "dataset": dataset, **summary})
    atomic_json(run_dir / "final_table.json", table)
    cols = [
        "method", "method_id", "dataset", "accuracy", "token_m", "token_scope",
        "input_tokens", "output_tokens", "total_tokens", "num_samples", "num_scored",
        "num_calls", "usage_sources", "end_to_end_token_m", "end_to_end_input_tokens",
        "end_to_end_output_tokens", "end_to_end_total_tokens", "end_to_end_num_calls",
        "end_to_end_usage_sources",
    ]
    with (run_dir / "final_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({
            **row,
            "token_m": _fmt_tokens(row["token_m"]),
            "end_to_end_token_m": _fmt_tokens(row["end_to_end_token_m"]),
        } for row in table)
    lines = [r"\begin{tabular}{llrr}", r"Method & Dataset & Accuracy & Selector Token (M) \\", r"\hline"]
    for x in table:
        lines.append(f"{_latex_escape(x['method'])} & {_latex_escape(x['dataset'])} & {_fmt_acc(x['accuracy'])} & {x['token_m']:.2f} " + r"\\")
    lines.append(r"\end{tabular}")
    (run_dir / "final_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_paper_tables(run_dir, table)
    return table
