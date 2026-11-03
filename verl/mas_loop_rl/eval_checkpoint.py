"""Resolve local Transformers checkpoints for MAS-Loop evaluation."""
from __future__ import annotations
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ResolvedCheckpoint:
    role: str
    source_type: str
    source: str
    mounted_path: str
    huggingface_path: str
    def to_dict(self) -> dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResolvedCheckpoint":
        required=("role","source_type","source","mounted_path","huggingface_path")
        missing=[k for k in required if not value.get(k)]
        if missing: raise ValueError(f"resolved checkpoint metadata is missing: {missing}")
        payload={k:value[k] for k in required}
        payload["huggingface_path"]=str(find_huggingface_checkpoint(payload["huggingface_path"]))
        return cls(**payload)

def _global_step(path: Path) -> int:
    vals=[int(x) for x in re.findall(r"(?:global_)?step_(\d+)", str(path))]
    return max(vals, default=-1)

def _is_hf_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path/"config.json").is_file() and any(p.is_file() for p in path.glob("*.safetensors"))

def find_huggingface_checkpoint(root: str|Path) -> Path:
    root=Path(os.path.abspath(os.path.expanduser(str(root))))
    if not root.exists(): raise FileNotFoundError(f"checkpoint path does not exist: {root}")
    candidates=[]
    if _is_hf_checkpoint(root): candidates.append(root)
    for config in root.rglob("config.json"):
        if _is_hf_checkpoint(config.parent): candidates.append(config.parent)
    if not candidates: raise FileNotFoundError(f"no Transformers checkpoint containing config.json and *.safetensors found under {root}")
    unique={str(p):p for p in candidates}
    return max(unique.values(), key=lambda p:(_global_step(p),-len(p.relative_to(root).parts),p.stat().st_mtime,str(p)))

def resolve_checkpoint(*, role: str, local_path: str|None=None) -> ResolvedCheckpoint:
    if not local_path: raise ValueError(f"{role}: local_path is required")
    mounted=Path(os.path.abspath(os.path.expanduser(local_path))); hf=find_huggingface_checkpoint(mounted)
    return ResolvedCheckpoint(role=role,source_type="local",source=str(local_path),mounted_path=str(mounted),huggingface_path=str(hf))
