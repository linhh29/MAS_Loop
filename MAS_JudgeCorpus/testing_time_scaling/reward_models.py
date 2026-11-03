from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardScorer:
    def __init__(self, *, base_model: Path, adapter: Path | None, max_length: int = 8192, batch_size: int = 2, device: str = "cuda"):
        self.base_model_path, self.adapter_path = Path(base_model), Path(adapter) if adapter else None
        self.max_length, self.batch_size = int(max_length), int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, local_files_only=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForSequenceClassification.from_pretrained(self.base_model_path, num_labels=1, torch_dtype=dtype, local_files_only=True, trust_remote_code=True, device_map="auto" if device == "cuda" else None)
        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, self.adapter_path, is_trainable=False)
        else:
            self.model = base
        self.model.eval()

    @torch.inference_mode()
    def score(self, texts: list[str]) -> tuple[list[float], dict[str, Any]]:
        scores, input_tokens, item_input_tokens = [], 0, []
        device = next(self.model.parameters()).device
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start+self.batch_size]
            tok = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            counts = [int(x) for x in tok["attention_mask"].sum(dim=1).tolist()]
            item_input_tokens.extend(counts)
            input_tokens += sum(counts)
            tok = {k:v.to(device) for k,v in tok.items()}
            logits = self.model(**tok).logits.float().reshape(-1)
            scores.extend(float(x) for x in logits.cpu().tolist())
        return scores, {"input_tokens":input_tokens,"output_tokens":0,"total_tokens":input_tokens,"usage_source":"tokenizer_exact","num_calls":len(texts),"model_kind":"sequence_classification","item_input_tokens":item_input_tokens}

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def rm_decisions(sample_rows: list[list[dict[str, Any]]], scorer: RewardScorer, *, trajectory: bool) -> list[dict[str, Any]]:
    """Score multiple samples in one forward batch: 4 samples x 16 candidates = 64 texts."""
    from .accounting import sum_usage
    from .selection import mas_generation_usage
    groups = [sorted(rows, key=lambda x: int(x["candidate_idx"])) for rows in sample_rows]
    key = "rm_text_trajectory" if trajectory else "rm_text_orchestration"
    flat = [x[key] for rows in groups for x in rows]
    scores, rm_usage = scorer.score(flat)
    per_item = rm_usage.get("item_input_tokens", [0] * len(flat))
    n = len(groups[0]) if groups else 0
    out = []
    for j, rows in enumerate(groups):
        local = scores[j*n:(j+1)*n]
        idx = max(range(len(local)), key=lambda i: (local[i], -i))
        item_tokens = per_item[j*n:(j+1)*n]
        usage = {"input_tokens": sum(item_tokens), "output_tokens": 0,
                 "total_tokens": sum(item_tokens), "usage_source": "tokenizer_exact",
                 "num_calls": len(rows), "model_kind": "sequence_classification"}
        out.append({"selected_index":idx,"prediction":rows[idx]["prediction"],"correct":bool(rows[idx]["correct"]),"scores":local,"usage":sum_usage([mas_generation_usage(rows),usage]),"selector_usage":usage})
    return out


def rm_decision(rows: list[dict[str, Any]], scorer: RewardScorer, *, trajectory: bool) -> dict[str, Any]:
    return rm_decisions([rows], scorer, trajectory=trajectory)[0]
