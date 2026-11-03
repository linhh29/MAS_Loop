from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUPERVISORS = (
    REPO_ROOT / "verl/scripts/run_bcp_only_supervised.sh",
    REPO_ROOT / "verl/scripts/run_deepscaler_only_supervised.sh",
)


def test_supervisor_monitor_owns_and_reaps_interval_sleep():
    for script in SUPERVISORS:
        text = script.read_text()
        assert 'local sleep_pid=""' in text, script
        assert 'sleep_pid=$!' in text, script
        assert 'kill "${sleep_pid}" 2>/dev/null || true' in text, script
        assert 'wait "${sleep_pid}" 2>/dev/null || true' in text, script


def test_supervisor_cleanup_reaps_both_background_workers():
    for script in SUPERVISORS:
        text = script.read_text()
        prefix = "BCP" if "bcp_only" in script.name else "DS"
        assert f'wait "${{{prefix}_MONITOR_PID}}" 2>/dev/null || true' in text, script
        assert f'wait "${{{prefix}_METRICS_PID}}" 2>/dev/null || true' in text, script
        assert "trap cleanup EXIT INT TERM" in text, script
