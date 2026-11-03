#!/usr/bin/env python3
"""Keep local Qwen3-8B busy while remote API selectors run."""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("KEEPALIVE_MODEL", "/path/to/models/Qwen3-8B")
DEVICE = os.environ.get("KEEPALIVE_DEVICE", "cuda:0")
DEVICE_MAP = os.environ.get("KEEPALIVE_DEVICE_MAP", "single")
INTERVAL = float(os.environ.get("KEEPALIVE_INTERVAL", "0"))
BATCH_SIZE = int(os.environ.get("KEEPALIVE_BATCH_SIZE", "128"))
INPUT_TOKENS = int(os.environ.get("KEEPALIVE_INPUT_TOKENS", "1024"))
MAX_NEW_TOKENS = int(os.environ.get("KEEPALIVE_MAX_NEW_TOKENS", "512"))
MIN_BATCH_SIZE = int(os.environ.get("KEEPALIVE_MIN_BATCH_SIZE", "1"))


def build_inputs(tokenizer, device: torch.device, batch_size: int) -> dict[str, torch.Tensor]:
    seed = (
        "Solve each task carefully and continue reasoning until token budget ends. "
        "Compare several approaches, check arithmetic, identify assumptions, and derive final answer. "
    )
    seed_ids = tokenizer(seed, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    repeats = (INPUT_TOKENS + seed_ids.numel() - 1) // seed_ids.numel()
    input_ids = seed_ids.repeat(repeats)[:INPUT_TOKENS].unsqueeze(0).repeat(batch_size, 1).to(device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if DEVICE_MAP == "auto" else {"": DEVICE},
    ).eval()
    input_device = model.get_input_embeddings().weight.device
    batch_size = BATCH_SIZE
    iteration = 0
    print(
        f"loaded model={MODEL} device_map={DEVICE_MAP} input_device={input_device} "
        f"batch={batch_size} input_tokens={INPUT_TOKENS} new_tokens={MAX_NEW_TOKENS}",
        flush=True,
    )
    with torch.inference_mode():
        while True:
            try:
                inputs = build_inputs(tokenizer, input_device, batch_size)
                started = time.monotonic()
                output = model.generate(
                    **inputs,
                    min_new_tokens=MAX_NEW_TOKENS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
                iteration += 1
                elapsed = time.monotonic() - started
                generated = output.shape[0] * (output.shape[1] - inputs["input_ids"].shape[1])
                print(
                    f"iteration={iteration} batch={batch_size} elapsed_s={elapsed:.2f} "
                    f"generated_tokens={generated} tokens_per_s={generated / elapsed:.1f}",
                    flush=True,
                )
                del inputs, output
                if INTERVAL > 0:
                    time.sleep(INTERVAL)
            except torch.cuda.OutOfMemoryError:
                if batch_size <= MIN_BATCH_SIZE:
                    raise
                batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                print(f"oom: reducing batch to {batch_size}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
