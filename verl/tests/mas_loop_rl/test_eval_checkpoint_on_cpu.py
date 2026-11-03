from pathlib import Path

from mas_loop_rl.eval_checkpoint import find_huggingface_checkpoint, resolve_checkpoint


def _fake_hf(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"checkpoint")


def test_resolver_preserves_lexical_symlink_prefix(tmp_path: Path) -> None:
    target = tmp_path / "target"
    hf = target / "round_1" / "orchestra" / "global_step_2" / "actor" / "huggingface"
    _fake_hf(hf)
    alias = tmp_path / "data_alias"
    alias.symlink_to(target, target_is_directory=True)

    resolved = resolve_checkpoint(role="orchestra", local_path=str(alias))

    assert resolved.source == str(alias)
    assert resolved.mounted_path == str(alias)
    assert resolved.huggingface_path.startswith(str(alias))
    assert resolved.huggingface_path.endswith("global_step_2/actor/huggingface")


def test_find_huggingface_checkpoint_selects_latest_step(tmp_path: Path) -> None:
    _fake_hf(tmp_path / "global_step_2" / "actor" / "huggingface")
    _fake_hf(tmp_path / "global_step_11" / "actor" / "huggingface")

    found = find_huggingface_checkpoint(tmp_path)

    assert "global_step_11" in str(found)
