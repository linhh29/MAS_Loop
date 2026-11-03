from pathlib import Path

from mas_loop_rl.analyze_vllm_concurrency_run import parse_runtime_metrics, parse_scheduler_log, parse_step_metrics


def test_parse_scheduler_distribution(tmp_path: Path):
    log = tmp_path / "vllm.log"
    log.write_text(
        "Running: 10 reqs, Waiting: 3 reqs\nGPU KV cache usage: 20.0%\n"
        "Running: 7 reqs, Waiting: 0 reqs\nGPU KV cache usage: 40.0%\n"
    )
    result = parse_scheduler_log(log)
    assert result["samples"] == 2
    assert result["kv_cache_pct"] == {"samples": 2, "mean": 30.0, "p50": 30.0, "p95": 39.0, "max": 40.0}
    assert result["running_max"] == 10
    assert result["waiting_max"] == 3


def test_parse_latest_step_metrics(tmp_path: Path):
    log = tmp_path / "full_train.log"
    log.write_text(
        "step:1 - timing_s/gen:12.5 - timing_s/step:20 - perf/throughput:4\n"
        "step:2 - timing_s/gen:10 - timing_s/step:18 - perf/throughput:5.5\n"
    )
    result = parse_step_metrics(log)
    assert result["timing_s/gen"]["values"] == [12.5, 10.0]
    assert result["timing_s/gen"]["mean"] == 11.25
    assert result["timing_s/step"]["values"] == [20.0, 18.0]
    assert result["perf/throughput"]["values"] == [4.0, 5.5]


def test_parse_scheduler_reports_pool_and_absolute_kv(tmp_path: Path):
    log = tmp_path / "vllm.log"
    log.write_text(
        "Available KV cache memory: 80.00 GiB\n"
        "Available KV cache memory: 79.00 GiB\n"
        "GPU KV cache size: 4,600,000 tokens\n"
        "GPU KV cache size: 4,500,000 tokens\n"
        "Maximum concurrency for 8,192 tokens per request: 555.42x\n"
        "GPU KV cache usage: 50.0%\n"
        "GPU KV cache usage: 70.0%\n"
    )
    result = parse_scheduler_log(log)
    assert result["kv_pool"] == {
        "rank_samples": 2,
        "gib_per_gpu_mean": 79.5,
        "gib_per_gpu_min": 79.0,
        "gib_per_gpu_max": 80.0,
        "tokens_per_gpu_mean": 4550000.0,
        "target_70pct_gib_per_gpu": 55.65,
    }
    assert result["active_kv_gib_per_gpu"] == {
        "mean": 47.7,
        "p50": 47.7,
        "p95": 54.855,
        "max": 55.65,
    }
    assert result["maximum_concurrency"] == {
        "tokens_per_request": 8192,
        "concurrency_x": 555.42,
    }


def test_parse_runtime_normalizes_kv_fraction_to_percent(tmp_path: Path):
    runtime = tmp_path / "runtime_metrics.jsonl"
    runtime.write_text(
        '{"time": 1, "gpus": [], "vllm": {"editor": {'
        '"vllm:kv_cache_usage_perc": 0.5, "vllm:num_requests_running": 10}}, '
        '"dispatch": {}, "coordinator": {}}\n'
        '{"time": 3, "gpus": [], "vllm": {"editor": {'
        '"vllm:kv_cache_usage_perc": 0.9, "vllm:num_requests_running": 20}}, '
        '"dispatch": {}, "coordinator": {}}\n'
    )
    result = parse_runtime_metrics(runtime)
    assert result["vllm"]["editor"]["kv_cache_usage_pct"] == {
        "samples": 2, "mean": 70.0, "p50": 70.0, "p95": 88.0, "max": 90.0
    }


def test_parse_scheduler_aggregates_data_parallel_ranks(tmp_path: Path):
    log = tmp_path / "vllm-dp.log"
    log.write_text(
        "INFO 08-10 10:50:25 Engine 000: Running: 37 reqs, Waiting: 76 reqs, GPU KV cache usage: 17.6%\n"
        "INFO 08-10 10:50:25 Engine 001: Running: 38 reqs, Waiting: 75 reqs, GPU KV cache usage: 18.8%\n"
        "INFO 08-10 10:50:35 Engine 000: Running: 71 reqs, Waiting: 42 reqs, GPU KV cache usage: 39.1%\n"
        "INFO 08-10 10:50:35 Engine 001: Running: 73 reqs, Waiting: 41 reqs, GPU KV cache usage: 40.3%\n"
    )
    result = parse_scheduler_log(log)
    assert result["rank_count"] == 2
    assert result["running_per_rank_peak"] == {"0": 71, "1": 73}
    assert result["waiting_per_rank_peak"] == {"0": 76, "1": 75}
    assert result["running_global_sum_peak"] == 144
    assert result["waiting_global_sum_peak"] == 151
    assert result["running_max"] == 144
    assert result["waiting_max"] == 151
