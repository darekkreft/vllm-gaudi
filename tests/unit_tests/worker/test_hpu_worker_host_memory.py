# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HPUWorker host-memory teardown on swap/shutdown."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_gaudi.v1.worker import hpu_worker as hpu_worker_mod


class _FakeRunner:

    def __init__(self, name: str):
        self.model = MagicMock()
        self.model.parameters.return_value = []
        self.model.buffers.return_value = []
        self.kv_caches = ["block"]
        self.defragmenter = object()
        self.vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={}),
            model_config=SimpleNamespace(model=name),
        )
        self.shutdown_inc = MagicMock()


def _make_worker() -> hpu_worker_mod.HPUWorker:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="m",
            dtype="bfloat16",
            enforce_eager=False,
            max_model_len=8192,
            seed=0,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=128),
        lora_config=None,
        load_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(rank=0, world_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        device_config=SimpleNamespace(device="hpu"),
        speculative_config=None,
        observability_config=None,
        compilation_config=SimpleNamespace(
            compile_ranges_endpoints=(),
            compile_sizes=(),
            static_forward_context={},
        ),
    )
    worker = hpu_worker_mod.HPUWorker.__new__(hpu_worker_mod.HPUWorker)
    worker._apply_vllm_config(vllm_config)
    worker.model_runner = None
    worker.model_sleeping = False
    worker.kv_cache_sleeping = False
    worker.kv_cache_config = None
    worker._model_runner_stash = {}
    worker._model_runner_state_stash = {}
    return worker


def test_evict_stashed_runners_drops_other_models() -> None:
    worker = _make_worker()
    keep = _FakeRunner("keep")
    drop = _FakeRunner("drop")
    worker._model_runner_stash = {("keep",): keep, ("drop",): drop}
    worker._model_runner_state_stash = {
        ("keep",): {"vllm_config": keep.vllm_config},
        ("drop",): {"vllm_config": drop.vllm_config},
    }

    with patch.object(hpu_worker_mod, "_trim_host_memory"):
        worker._evict_stashed_runners(except_key=("keep",))

    assert ("keep",) in worker._model_runner_stash
    assert ("drop",) not in worker._model_runner_stash
    assert drop.model is None
    drop.shutdown_inc.assert_called()


def test_shutdown_destroy_active_and_stashed() -> None:
    worker = _make_worker()
    active = _FakeRunner("active")
    stashed = _FakeRunner("stashed")
    worker.model_runner = active
    worker._model_runner_stash = {("stashed",): stashed}

    with patch.object(hpu_worker_mod, "_trim_host_memory"):
        with patch.object(hpu_worker_mod.HPUBucketingManager, "deactivate"):
            worker.shutdown()

    assert worker.model_runner is None
    assert worker._model_runner_stash == {}
    assert active.model is None
    assert stashed.model is None


def test_trim_host_memory_runs_gc_malloc_trim_and_tcmalloc_release() -> None:
    mock_libc = MagicMock()
    mock_tc = MagicMock()

    def _cdll(name: str) -> MagicMock:
        if "tcmalloc" in name:
            return mock_tc
        return mock_libc

    with patch.object(hpu_worker_mod.gc, "collect") as collect:
        with patch("ctypes.CDLL", side_effect=_cdll):
            with patch.object(hpu_worker_mod, "_trim_hpu_device_memory") as trim_hpu:
                hpu_worker_mod._trim_host_memory()

    assert collect.call_count == 2
    mock_libc.malloc_trim.assert_called_once_with(0)
    mock_tc.MallocExtension_ReleaseFreeMemory.assert_called_once_with()
    trim_hpu.assert_called_once()


def test_load_model_evicts_stash_before_new_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _make_worker()
    stashed = _FakeRunner("old")
    worker._model_runner_stash = {("old",): stashed}

    new_config = worker.vllm_config
    new_config.model_config.model = "new-checkpoint"

    monkeypatch.setattr(hpu_worker_mod, "set_current_vllm_config", lambda cfg: contextlib_null(cfg))
    monkeypatch.setattr(hpu_worker_mod, "HPUModelRunner", lambda **kwargs: _FakeRunner("new"))

    with patch.object(worker, "_evict_stashed_runners") as evict:
        with patch.object(worker, "_runner_stash_key", return_value=("missing",)):
            worker.load_model(vllm_config=new_config)

    evict.assert_called_once_with()


class contextlib_null:
    def __init__(self, _cfg):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
