from mas_loop_rl.wait_for_gpu_isolation import _parse_visible_gpus


def test_parse_numeric_cuda_visibility():
    env = b"PATH=/bin\0CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\0"
    assert _parse_visible_gpus(env) == tuple(range(8))


def test_parse_empty_visibility_means_no_gpu():
    assert _parse_visible_gpus(b"CUDA_VISIBLE_DEVICES=\0") == ()
    assert _parse_visible_gpus(b"CUDA_VISIBLE_DEVICES=-1\0") == ()


def test_parse_unspecified_or_uuid_visibility_fails_closed():
    assert _parse_visible_gpus(b"PATH=/bin\0") is None
    assert _parse_visible_gpus(b"CUDA_VISIBLE_DEVICES=GPU-deadbeef\0") is None


def test_disjoint_numeric_visibility_can_be_checked_exactly():
    visible = set(_parse_visible_gpus(b"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\0") or ())
    assert visible.isdisjoint(set(range(8, 16)))
