import json
import time

from mas_loop_rl.monitor_runtime_state import _ray_lb_snapshot


def test_ray_lb_snapshot_uses_newest_global_state_without_summing(tmp_path):
    now = time.time()
    (tmp_path / "ray-lb-1.json").write_text(json.dumps({
        "time": now - 1,
        "pid": 1,
        "total_inflight": 7,
        "max_outstanding": 8,
        "admission_waiters": 3,
        "admission_peak": 8,
        "acquire_wait_sec": 2.5,
    }))
    (tmp_path / "ray-lb-2.json").write_text(json.dumps({
        "time": now,
        "pid": 2,
        "total_inflight": 6,
        "max_outstanding": 8,
        "admission_waiters": 1,
        "admission_peak": 8,
        "acquire_wait_sec": 0.5,
    }))

    result = _ray_lb_snapshot(tmp_path, stale_after_sec=30)
    assert result["reporters"] == 2
    assert result["pid"] == 2
    assert result["total_inflight"] == 6
    assert result["observed_max_total_inflight"] == 7
    assert result["observed_max_waiters"] == 3
    assert result["observed_max_acquire_wait_sec"] == 2.5
