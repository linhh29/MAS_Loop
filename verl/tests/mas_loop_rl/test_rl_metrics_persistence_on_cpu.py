import json
from pathlib import Path

from mas_loop_rl.rl_metrics import persist_rl_metrics


def test_persist_rl_metrics_appends_all_values_under_phase_ckpt(tmp_path: Path, monkeypatch):
    target = tmp_path / "round_1" / "orchestra" / "metrics" / "training_metrics.jsonl"
    monkeypatch.setenv("MAS_LOOP_RL_METRICS_FILE", str(target))
    monkeypatch.setenv("MAS_LOOP_DATASET_NAME", "bcp")
    monkeypatch.setenv("MAS_LOOP_ROLE", "orchestra")
    monkeypatch.setenv("MAS_LOOP_OUTER_ROUND", "1")

    persist_rl_metrics({"actor/loss": 1.25, "actor/ppo_kl": 0.02, "actor/grad_norm": 3}, 4)
    persist_rl_metrics({"critic/rewards/mean": -0.5, "training/global_step": 5}, 5)

    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["global_step"] for row in rows] == [4, 5]
    assert rows[0]["dataset"] == "bcp"
    assert rows[0]["role"] == "orchestra"
    assert rows[0]["metrics"]["actor/ppo_kl"] == 0.02
    assert rows[1]["metrics"]["critic/rewards/mean"] == -0.5
