from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = "/path/to/checkpoints"
SEARCH_CLIENTS = (
    "anthropic_client.py",
    "search_r1_client.py",
    "gemini_client.py",
    "glm_zai_client.py",
    "openai_client.py",
    "qwen_client.py",
    "tongyi_client.py",
)


def test_mas_loop_runtime_fallbacks_do_not_write_inside_repository() -> None:
    for relative in (
        "verl/mas_loop_rl/loop_agent.py",
        "verl/mas_loop_rl/executor.py",
        "verl/mas_loop_rl/reward.py",
    ):
        text = (REPO_ROOT / relative).read_text()
        assert CHECKPOINT_ROOT in text
        assert "verl/outputs/mas_loop_grpo" not in text
        assert "/" + "ossfs/" not in text
        assert f"{CHECKPOINT_ROOT}/work" not in text


def test_browsecomp_search_clients_default_to_shared_eval_storage() -> None:
    root = REPO_ROOT / "datasets/BrowserComp-Plus/search_agent"
    if not root.exists():
        root = REPO_ROOT / "datasets/BrowseComp-Plus/search_agent"
    for name in SEARCH_CLIENTS:
        text = (root / name).read_text()
        assert "MAS_LOOP_EVAL_OUTPUT_ROOT" in text
        assert f'{CHECKPOINT_ROOT}/eval_outputs' in text
        assert 'default="runs/' not in text


def test_checkpoint_matrix_default_is_outside_repository() -> None:
    text = (REPO_ROOT / "datasets/run_all_mas_loop_checkpoint_matrix_eval.sh").read_text()
    assert f"MAS_LOOP_MATRIX_OUTPUT_ROOT:-{CHECKPOINT_ROOT}/eval/" in text
    assert "$ROOT/datasets/eval_outputs" not in text
