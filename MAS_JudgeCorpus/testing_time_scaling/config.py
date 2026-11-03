from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out=dict(a)
    for k,v in b.items(): out[k]=_merge(out.get(k,{}) if isinstance(out.get(k),dict) else {},v) if isinstance(v,dict) else v
    return out


def _expand(value: Any) -> Any:
    if isinstance(value,dict): return {k:_expand(v) for k,v in value.items()}
    if isinstance(value,list): return [_expand(v) for v in value]
    return os.path.expandvars(value) if isinstance(value,str) else value


def load_config(path: Path, secrets_path: Path | None = None) -> dict[str, Any]:
    cfg=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if secrets_path and secrets_path.is_file(): cfg=_merge(cfg,yaml.safe_load(secrets_path.read_text(encoding="utf-8")) or {})
    cfg = _expand(cfg)
    # Runtime safety valve for provider TPM limits. Never raises configured RPM;
    # env cap only lowers it for this experiment process.
    try:
        rpm_cap = float(os.environ.get("MAS_JUDGE_RPM_CAP", "0"))
    except ValueError:
        rpm_cap = 0.0
    if rpm_cap > 0:
        for name in ("deepseek_v4_flash", "kimi_k3", "qwen3_8_2_4t_a95b"):
            if name in cfg.get("endpoints", {}):
                current = float(cfg["endpoints"][name].get("requests_per_minute", 0) or 0)
                if current <= 0 or current > rpm_cap:
                    cfg["endpoints"][name]["requests_per_minute"] = rpm_cap
    endpoints = cfg.setdefault("endpoints", {})
    # Accept original human-facing YAML key while keeping filesystem-safe method ID.
    for alias in ("qwen3.8-2.4t-a95b", "qwen3.8-2.4t", "qwen3_8_2_4t"):
        if alias in endpoints and "qwen3_8_2_4t_a95b" not in endpoints:
            endpoints["qwen3_8_2_4t_a95b"] = endpoints.pop(alias)
    return cfg
