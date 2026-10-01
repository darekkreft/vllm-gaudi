# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cloudpickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
import yaml

from vllm_gaudi.v1.engine.multi_model_async_llm import MultiModelAsyncLLM
from vllm_gaudi.entrypoints.openai import multi_model_api_server as api_server
from vllm_gaudi.v1.engine import core_patch


class _FakeAsyncEngineArgs:

    def __init__(self, model: str, max_model_len: int = 4096, **_: object):
        self.model = model
        self.max_model_len = max_model_len
        self.disable_log_stats = False
        self.enable_log_requests = False

    def create_engine_config(self, _usage_context):
        return SimpleNamespace(model_config=SimpleNamespace(
            model=self.model,
            max_model_len=self.max_model_len,
        ))


class _FakeVllmConfig:

    def __init__(self):
        self.model_config = SimpleNamespace(model="test-model", runner_type="generate")
        self.cache_config = SimpleNamespace()
        self.scheduler_config = SimpleNamespace()
        self.parallel_config = SimpleNamespace()


@pytest.fixture
def mock_engine():
    engine = AsyncMock()
    engine.wait_for_requests_to_drain = AsyncMock()
    engine.wake_up = AsyncMock()
    engine.shutdown = Mock()
    engine.engine_core = SimpleNamespace(call_utility_async=AsyncMock())
    return engine


def test_load_multi_model_config_success(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "default_model": "llama",
            "models": {
                "llama": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "max_model_len": 4096,
                },
                "qwen": {
                    "model": "Qwen/Qwen3-0.6B",
                    "max_model_len": 4096,
                },
            },
        }))

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        config = api_server._load_multi_model_config(str(cfg_path))

    assert config.default_model == "llama"
    assert set(config.model_configs.keys()) == {"llama", "qwen"}
    assert config.model_frontend_overrides == {
        "llama": api_server.ModelFrontendOverrides(),
        "qwen": api_server.ModelFrontendOverrides(),
    }
    assert config.model_quant_configs == {}


def test_load_multi_model_config_falls_back_to_model_env(tmp_path, monkeypatch):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "models": {
                "llama": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                },
                "qwen": {
                    "model": "Qwen/Qwen3-0.6B",
                },
            },
        }))
    monkeypatch.setenv("MODEL", "qwen")

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        config = api_server._load_multi_model_config(str(cfg_path))

    assert config.default_model == "qwen"


def test_load_multi_model_config_extracts_frontend_overrides(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "default_model": "a",
            "models": {
                "a": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "max_model_len": 4096,
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "granite",
                    "chat_template": "./templates/tool_a.jinja",
                },
                "b": {
                    "model": "Qwen/Qwen3-0.6B",
                    "max_model_len": 8192,
                },
            },
        }))

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        config = api_server._load_multi_model_config(str(cfg_path))

    assert set(config.model_configs.keys()) == {"a", "b"}
    assert config.model_quant_configs == {}
    assert config.model_frontend_overrides["a"] == api_server.ModelFrontendOverrides(
        enable_auto_tool_choice=True,
        tool_call_parser="granite",
        chat_template=str((tmp_path / "templates" / "tool_a.jinja").resolve()),
    )
    assert config.model_frontend_overrides["b"] == api_server.ModelFrontendOverrides()


def test_load_multi_model_config_extracts_quant_configs(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "default_model": "quantized",
            "models": {
                "quantized": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "max_model_len": 4096,
                    "quantization": "inc",
                    "quant_config": "quant/maxabs.json",
                },
                "plain": {
                    "model": "Qwen/Qwen3-0.6B",
                    "max_model_len": 4096,
                },
            },
        }))

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        config = api_server._load_multi_model_config(str(cfg_path))

    expected_path = str((tmp_path / "quant" / "maxabs.json").resolve())
    assert config.model_quant_configs["quantized"] == expected_path
    assert "plain" not in config.model_quant_configs


def test_load_multi_model_config_extracts_explicit_quant_config_null(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "default_model": "a",
            "models": {
                "a": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "quant_config": None,
                },
                "b": {
                    "model": "Qwen/Qwen3-0.6B",
                },
            },
        }))

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        config = api_server._load_multi_model_config(str(cfg_path))

    assert config.model_quant_configs == {"a": None}


def test_resolve_frontend_settings_uses_model_override_then_cli():
    args = SimpleNamespace(
        enable_auto_tool_choice=False,
        tool_call_parser="hermes",
        chat_template="/global/template.jinja",
    )
    model_frontend_overrides = {
        "a":
        api_server.ModelFrontendOverrides(
            enable_auto_tool_choice=True,
            tool_call_parser="granite",
            chat_template="/model/a_template.jinja",
        ),
        "b":
        api_server.ModelFrontendOverrides(tool_call_parser="mistral", ),
    }

    settings_a = api_server._resolve_frontend_settings(args, model_frontend_overrides, "a")
    settings_b = api_server._resolve_frontend_settings(args, model_frontend_overrides, "b")

    assert settings_a == api_server.FrontendSettings(
        enable_auto_tool_choice=True,
        tool_call_parser="granite",
        chat_template="/model/a_template.jinja",
    )
    assert settings_b == api_server.FrontendSettings(
        enable_auto_tool_choice=False,
        tool_call_parser="mistral",
        chat_template="/global/template.jinja",
    )


@pytest.mark.asyncio
async def test_initialize_and_switch_reconfigures_engine(mock_engine):
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
        "qwen": _FakeAsyncEngineArgs("Qwen/Qwen3-0.6B"),
    }

    with patch("vllm_gaudi.v1.engine.multi_model_async_llm.AsyncLLM.from_engine_args",
               return_value=mock_engine), patch.object(MultiModelAsyncLLM,
                                                       "_refresh_engine_frontend_config",
                                                       new_callable=AsyncMock):
        manager = MultiModelAsyncLLM(model_configs, model_quant_configs={"qwen": "/tmp/qwen_quant.json"})
        await manager.initialize("llama")
        await manager.switch_model("qwen", drain_timeout=1)

    assert manager.current_model == "qwen"
    mock_engine.wait_for_requests_to_drain.assert_awaited_once()
    mock_engine.engine_core.call_utility_async.assert_awaited_once_with(
        "gaudi_reconfigure_engine",
        cloudpickle.dumps(manager.get_vllm_config("qwen")),
        "/tmp/qwen_quant.json",
    )


@pytest.mark.asyncio
async def test_switch_preserves_quant_config_when_not_specified(mock_engine):
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
        "qwen": _FakeAsyncEngineArgs("Qwen/Qwen3-0.6B"),
    }

    with patch("vllm_gaudi.v1.engine.multi_model_async_llm.AsyncLLM.from_engine_args",
               return_value=mock_engine), patch.object(MultiModelAsyncLLM,
                                                       "_refresh_engine_frontend_config",
                                                       new_callable=AsyncMock):
        manager = MultiModelAsyncLLM(model_configs, model_quant_configs={"llama": "/tmp/llama_quant.json"})
        await manager.initialize("llama")
        await manager.switch_model("qwen", drain_timeout=1)

    mock_engine.engine_core.call_utility_async.assert_awaited_once_with(
        "gaudi_reconfigure_engine",
        cloudpickle.dumps(manager.get_vllm_config("qwen")),
    )


@pytest.mark.asyncio
async def test_switch_same_model_is_noop(mock_engine):
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
    }

    with patch("vllm_gaudi.v1.engine.multi_model_async_llm.AsyncLLM.from_engine_args", return_value=mock_engine):
        manager = MultiModelAsyncLLM(model_configs)
        await manager.initialize("llama")
        await manager.switch_model("llama")

    mock_engine.engine_core.call_utility_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_invalid_model_raises():
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
    }
    manager = MultiModelAsyncLLM(model_configs)

    with pytest.raises(ValueError, match="not found"):
        await manager.initialize("qwen")


def test_deserialize_reconfigure_config_requires_insecure_serialization(monkeypatch):
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", False)

    payload = cloudpickle.dumps(_FakeVllmConfig())

    with pytest.raises(RuntimeError, match="VLLM_ALLOW_INSECURE_SERIALIZATION=1"):
        core_patch._deserialize_reconfigure_config(payload)


def test_deserialize_reconfigure_config_rejects_non_vllm_config(monkeypatch):
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", True)

    payload = cloudpickle.dumps({"model": "not-a-config"})

    with pytest.raises(TypeError, match="expected VllmConfig"):
        core_patch._deserialize_reconfigure_config(payload)


def test_deserialize_reconfigure_config_error_does_not_leak_payload(monkeypatch):
    """
    Error/log leak scan: when a reconfigure payload is rejected, the raised
    error must not echo the (untrusted, possibly secret-bearing) payload back.
    """
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", True)

    secret = "SUPER_SECRET_TOKEN_1234"
    payload = cloudpickle.dumps({"model": secret})
    # Sanity-check the secret is actually in the payload, so the leak
    # assertion below is meaningful and not a false negative.
    assert secret.encode() in payload

    with pytest.raises(TypeError) as exc_info:
        core_patch._deserialize_reconfigure_config(payload)

    assert secret not in str(exc_info.value)


def test_deserialize_reconfigure_config_accepts_valid_payload(monkeypatch):
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", True)

    expected = _FakeVllmConfig()
    payload = cloudpickle.dumps(expected)

    decoded = core_patch._deserialize_reconfigure_config(payload)

    assert isinstance(decoded, _FakeVllmConfig)
    assert decoded.model_config.model == "test-model"


_GIB = 2**30


def test_host_headroom_guard_rejects_sleep_when_cgroup_is_too_small(monkeypatch):
    monkeypatch.delenv("VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD", raising=False)
    reports = [
        {"required_bytes": 40 * _GIB, "headroom_bytes": 70 * _GIB},
        {"required_bytes": 40 * _GIB, "headroom_bytes": 60 * _GIB},
    ]

    with pytest.raises(RuntimeError, match="insufficient host memory to sleep"):
        core_patch._ensure_host_headroom_for_sleep(reports)


@pytest.mark.parametrize(
    "reports",
    [
        [{"required_bytes": 40 * _GIB, "headroom_bytes": 60 * _GIB}],
        [{"required_bytes": 40 * _GIB, "headroom_bytes": None}],
        [{"required_bytes": 0, "headroom_bytes": 0}],
        None,
        [object()],
    ],
)
def test_host_headroom_guard_allows_sleep(monkeypatch, reports):
    monkeypatch.delenv("VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD", raising=False)
    core_patch._ensure_host_headroom_for_sleep(reports)


def test_host_headroom_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD", "1")
    core_patch._ensure_host_headroom_for_sleep([{"required_bytes": 2 * _GIB, "headroom_bytes": _GIB}])


def test_gaudi_reconfigure_engine_keeps_current_model_when_host_headroom_is_insufficient(monkeypatch):
    monkeypatch.delenv("VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD", raising=False)

    class _FakeModelExecutor:

        def __init__(self):
            self.is_sleeping = False
            self.sleep_called = False

        def sleep(self, level: int = 1):
            self.sleep_called = True

    class _FakeEngineCore:

        def __init__(self):
            self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="old-model"))
            self.model_executor = _FakeModelExecutor()
            self.rpc_calls: list[str] = []
            self.paused = False
            self.resume_scheduler_calls = 0

        def pause_scheduler(self, mode: str, clear_cache: bool):
            self.paused = True

        def collective_rpc(self, method: str, kwargs=None):
            self.rpc_calls.append(method)
            if method == "get_hpu_used_memory_mb":
                return [{"used": 100.0}]
            if method == "check_sleep_host_headroom":
                return [{"required_bytes": 64 * _GIB, "headroom_bytes": 10 * _GIB}]
            raise AssertionError(f"Unexpected RPC method: {method}")

        def resume_scheduler(self):
            self.resume_scheduler_calls += 1

    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config",
                        lambda _: SimpleNamespace(model_config=SimpleNamespace(model="new-model")))
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", Mock())

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="insufficient host memory to sleep"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    assert not fake_core.paused
    assert not fake_core.model_executor.sleep_called
    assert not {"unload_model", "load_model", "restore_stashed_model"} & set(fake_core.rpc_calls)
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_worker_sleep_host_guard_raises_before_moving_model(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_worker as hw

    monkeypatch.delenv("VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD", raising=False)
    monkeypatch.setattr(hw, "_host_cgroup_headroom_bytes", lambda: 10 * _GIB)
    monkeypatch.setattr(hw, "_device_resident_model_bytes", lambda _model: 8 * _GIB)

    worker = hw.HPUWorker.__new__(hw.HPUWorker)
    worker.model_sleeping = False
    worker.model_runner = SimpleNamespace(model=object())
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))

    with pytest.raises(RuntimeError, match="insufficient host memory to move the model to CPU"):
        worker._ensure_sleep_host_headroom()

    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=1))
    worker._ensure_sleep_host_headroom()


def test_worker_reports_no_host_requirement_when_model_already_sleeping(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_worker as hw

    monkeypatch.setattr(hw, "_host_cgroup_headroom_bytes", lambda: 5 * _GIB)
    worker = hw.HPUWorker.__new__(hw.HPUWorker)
    worker.model_sleeping = True
    worker.model_runner = SimpleNamespace(model=object())

    assert worker.check_sleep_host_headroom() == {"required_bytes": 0, "headroom_bytes": 5 * _GIB}


def test_host_cgroup_headroom_excludes_page_cache(monkeypatch, tmp_path):
    from vllm_gaudi.v1.worker import hpu_worker as hw

    (tmp_path / "memory.max").write_text(str(100 * _GIB))
    (tmp_path / "memory.current").write_text(str(70 * _GIB))
    (tmp_path / "memory.stat").write_text(f"anon {20 * _GIB}\nfile {50 * _GIB}\n")
    real_path = hw.Path
    monkeypatch.setattr(hw, "Path", lambda p: tmp_path if p == "/sys/fs/cgroup" else real_path(p))

    assert hw._host_cgroup_headroom_bytes() == 80 * _GIB

    (tmp_path / "memory.max").write_text("max")
    assert hw._host_cgroup_headroom_bytes() is None


def test_release_hpu_graph_wrapper_after_cpu_sleep_keeps_runner_usable(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_worker as hw

    adapter = object()
    graphed = {"bucket"}
    deactivate = Mock()
    monkeypatch.setattr(hw.HPUBucketingManager, "deactivate", deactivate)
    runner = SimpleNamespace(model=adapter, graphed_buckets=graphed)

    hw._release_hpu_graph_wrapper_after_cpu_sleep(runner)

    assert runner.model is adapter
    assert runner.graphed_buckets == {"bucket"}
    deactivate.assert_not_called()


def test_release_runner_host_memory_clears_weights_without_finalizing_by_default():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    weight = torch.nn.Parameter(torch.ones(2))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    runner = SimpleNamespace(kv_caches=[object()], defragmenter=object(), model=model)

    hw._release_runner_host_memory(runner, finalize_inc=False)

    assert runner.kv_caches == []
    assert runner.defragmenter is None
    assert runner.model is None
    assert weight.numel() == 0


def test_release_runner_host_memory_finalizes_inc_only_when_requested():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    calls: list[str] = []
    runner = SimpleNamespace(model=None, shutdown_inc=lambda: calls.append("shutdown_inc"))

    hw._release_runner_host_memory(runner, finalize_inc=False)
    assert calls == []

    hw._release_runner_host_memory(runner, finalize_inc=True)
    assert calls == ["shutdown_inc"]


def test_release_runner_host_memory_finalizes_inc_before_releasing_weights():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    weight = torch.nn.Parameter(torch.ones(4))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    seen: list[tuple[object, int]] = []
    runner = SimpleNamespace(model=model)
    runner.shutdown_inc = lambda: seen.append((runner.model, weight.numel()))

    hw._release_runner_host_memory(runner, finalize_inc=True)

    assert seen == [(model, 4)]
    assert weight.numel() == 0
    assert runner.model is None


def test_release_runner_host_memory_finalizes_inc_with_fallback_model_when_runner_model_cleared():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(4)))
    seen: list[object] = []
    runner = SimpleNamespace(model=None)
    runner.shutdown_inc = lambda: seen.append(runner.model)

    hw._release_runner_host_memory(runner, finalize_inc=True, fallback_model=model)

    assert seen == [model]
    assert model.weight.numel() == 0
    assert runner.model is None


def test_worker_shutdown_at_process_exit_only_finalizes_inc(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_worker as hw

    monkeypatch.setattr(hw, "_process_is_exiting", lambda: True)
    release = Mock()
    trim = Mock()
    monkeypatch.setattr(hw, "_release_runner_host_memory", release)
    monkeypatch.setattr(hw, "_trim_host_python_allocator", trim)
    worker = hw.HPUWorker.__new__(hw.HPUWorker)
    worker._model_runner_stash = {}
    worker._model_runner_state_stash = {}
    worker._loaded_model_ref = object()
    calls: list[str] = []
    worker.model_runner = SimpleNamespace(model=object(), shutdown_inc=lambda: calls.append("shutdown_inc"))

    worker.shutdown()

    assert calls == ["shutdown_inc"]
    release.assert_not_called()
    trim.assert_not_called()
    assert worker._loaded_model_ref is None


def test_worker_shutdown_releases_runner_host_memory_and_trims_allocator():
    from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

    worker = HPUWorker.__new__(HPUWorker)
    worker._model_runner_stash = {}
    worker._model_runner_state_stash = {}
    worker._loaded_model_ref = None
    calls: list[str] = []
    worker.model_runner = SimpleNamespace(model=None, shutdown_inc=lambda: calls.append("shutdown_inc"))

    worker.shutdown()

    assert calls == ["shutdown_inc"]
    assert worker._loaded_model_ref is None


def test_gaudi_reconfigure_engine_rolls_back_on_load_failure(monkeypatch):

    class _FakeNewConfig:

        def __init__(self):
            self.model_config = SimpleNamespace(model="new-model")

    class _FakeModelExecutor:

        def __init__(self):
            self.is_sleeping = False

        def sleep(self, level: int = 1):
            assert level == 1

    class _FakeEngineCore:

        def __init__(self):
            self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="old-model"))
            self.model_executor = _FakeModelExecutor()
            self.resume_scheduler_calls = 0
            self.restore_called = False
            self.used_memory_mb = 100.0

        def pause_scheduler(self, mode: str, clear_cache: bool):
            assert mode == "abort"
            assert clear_cache

        def collective_rpc(self, method: str, kwargs=None):
            if method == "get_hpu_used_memory_mb":
                return [{"used": self.used_memory_mb}]
            if method == "unload_model":
                self.used_memory_mb = 10.0
                return [{"stash_memory_after_mb": 7.0}]
            if method == "load_model":
                raise RuntimeError("load failed")
            if method == "restore_stashed_model":
                assert kwargs is not None
                assert kwargs["vllm_config"] is self.vllm_config
                assert kwargs["restore_kv_cache"] is True
                self.restore_called = True
                return [{"restored": True}]
            raise AssertionError(f"Unexpected RPC method: {method}")

        def resume_scheduler(self):
            self.resume_scheduler_calls += 1

    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config", lambda _: _FakeNewConfig())
    normalize_config = Mock()
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", normalize_config)

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="load failed"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    normalize_config.assert_called_once()
    assert normalize_config.call_args.args[0].model_config.model == "new-model"
    assert fake_core.restore_called is True
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_gaudi_reconfigure_engine_rolls_back_on_normalize_failure(monkeypatch):

    class _FakeNewConfig:

        def __init__(self):
            self.model_config = SimpleNamespace(model="new-model")

    class _FakeEngineCore:

        def __init__(self):
            self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="old-model"))
            self.resume_scheduler_calls = 0
            self.restore_called = False
            self.unload_called = False

        def collective_rpc(self, method: str, kwargs=None):
            if method == "unload_model":
                self.unload_called = True
                return [{"stash_memory_after_mb": 7.0}]
            if method == "restore_stashed_model":
                assert kwargs is not None
                assert kwargs["vllm_config"] is self.vllm_config
                self.restore_called = True
                return [{"restored": True}]
            raise AssertionError(f"Unexpected RPC method: {method}")

        def resume_scheduler(self):
            self.resume_scheduler_calls += 1

    def _raise_normalize_failure(_config):
        raise RuntimeError("normalize failed")

    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config", lambda _: _FakeNewConfig())
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", _raise_normalize_failure)

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="normalize failed"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    assert fake_core.unload_called is False
    assert fake_core.restore_called is False
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_gaudi_reconfigure_engine_skips_restore_without_stash_marker(monkeypatch):

    class _FakeNewConfig:

        def __init__(self):
            self.model_config = SimpleNamespace(model="new-model")

    class _FakeModelExecutor:

        def __init__(self):
            self.is_sleeping = False

        def sleep(self, level: int = 1):
            assert level == 1

    class _FakeEngineCore:

        def __init__(self):
            self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="old-model"))
            self.model_executor = _FakeModelExecutor()
            self.resume_scheduler_calls = 0
            self.restore_called = False
            self.used_memory_mb = 100.0

        def pause_scheduler(self, mode: str, clear_cache: bool):
            assert mode == "abort"
            assert clear_cache

        def collective_rpc(self, method: str, kwargs=None):
            if method == "get_hpu_used_memory_mb":
                return [{"used": self.used_memory_mb}]
            if method == "unload_model":
                self.used_memory_mb = 10.0
                return []
            if method == "load_model":
                raise RuntimeError("load failed")
            if method == "restore_stashed_model":
                self.restore_called = True
                return [{"restored": True}]
            raise AssertionError(f"Unexpected RPC method: {method}")

        def resume_scheduler(self):
            self.resume_scheduler_calls += 1

    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config", lambda _: _FakeNewConfig())
    normalize_config = Mock()
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", normalize_config)

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="load failed"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    normalize_config.assert_called_once()
    assert fake_core.restore_called is False
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_normalize_reconfigure_config_aligns_granite_hybrid_mamba_state(monkeypatch):

    class _FakeModelClass:

        @staticmethod
        def get_mamba_state_shape_from_config(_config):
            return [(1, )]

        @staticmethod
        def get_mamba_state_dtype_from_config(_config):
            return [torch.uint8]

    class _FakeMambaSpec:

        def __init__(self, **_kwargs):
            # Use a raw fallback value; platform hooks own any later padding/alignment.
            self.page_size_bytes = 2111

    model_config = SimpleNamespace(
        model="ibm-granite/granite-4.0-h-small",
        is_hybrid=True,
        architecture="GraniteMoeHybridForCausalLM",
        hf_config=SimpleNamespace(model_type="granitemoehybrid"),
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda _parallel: 1,
        get_head_size=lambda: 1,
    )
    cache_config = SimpleNamespace(
        block_size=528,
        mamba_block_size=None,
        mamba_cache_mode="none",
        mamba_page_size_padded=None,
        cache_dtype="auto",
    )
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    check_and_update_config = Mock()
    update_block_size_for_backend = Mock()
    monkeypatch.setattr(core_patch.current_platform, "check_and_update_config", check_and_update_config)
    monkeypatch.setattr(core_patch.current_platform, "update_block_size_for_backend", update_block_size_for_backend)
    monkeypatch.setattr(core_patch, "MambaSpec", _FakeMambaSpec)
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        lambda *_args, **_kwargs: (_FakeModelClass, None),
    )

    core_patch._normalize_reconfigure_config_for_platform(config)

    assert check_and_update_config.call_count == 2
    assert update_block_size_for_backend.call_count == 2
    assert cache_config.mamba_cache_mode == "align"
    assert cache_config.mamba_block_size == 528
    assert cache_config.mamba_page_size_padded == 2111


def test_normalize_reconfigure_resets_mamba_block_size_from_max_model_len_sentinel(monkeypatch):
    """Regression: _normalize_reconfigure_config_for_platform must reset
    mamba_block_size when it equals max_model_len (the 'none' mode sentinel).

    In vLLM's HybridAttentionMambaModelConfig.verify_and_update_config, when
    prefix caching is disabled (mamba_cache_mode='none'), mamba_block_size is
    set to max_model_len as a sentinel value meaning "one block per sequence".
    When the reconfigure path changes mamba_cache_mode to 'align', the old
    sentinel must be replaced with cache_config.block_size.  Without the fix,
    the KVCacheCoordinatorBase assertion (scheduler_block_size %
    mamba_block_size == 0) fails because max_model_len (e.g. 32768) is not
    divisible by the HPU attention block size (e.g. 528).
    """

    class _FakeModelClass:

        @staticmethod
        def get_mamba_state_shape_from_config(_config):
            return [(1, )]

        @staticmethod
        def get_mamba_state_dtype_from_config(_config):
            return [torch.uint8]

    class _FakeMambaSpec:

        def __init__(self, **_kwargs):
            self.page_size_bytes = 2111

    MAX_MODEL_LEN = 32768

    model_config = SimpleNamespace(
        model="ibm-granite/granite-4.0-h-small",
        is_hybrid=True,
        architecture="GraniteMoeHybridForCausalLM",
        hf_config=SimpleNamespace(model_type="granitemoehybrid"),
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda _parallel: 1,
        get_head_size=lambda: 1,
        max_model_len=MAX_MODEL_LEN,
    )
    cache_config = SimpleNamespace(
        block_size=528,
        mamba_block_size=MAX_MODEL_LEN,  # sentinel set by vLLM for "none" mode
        mamba_cache_mode="none",
        mamba_page_size_padded=2162688,
        cache_dtype="auto",
    )
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    check_and_update_config = Mock()
    update_block_size_for_backend = Mock()
    monkeypatch.setattr(core_patch.current_platform, "check_and_update_config", check_and_update_config)
    monkeypatch.setattr(core_patch.current_platform, "update_block_size_for_backend", update_block_size_for_backend)
    monkeypatch.setattr(core_patch, "MambaSpec", _FakeMambaSpec)
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        lambda *_args, **_kwargs: (_FakeModelClass, None),
    )

    core_patch._normalize_reconfigure_config_for_platform(config)

    # Mode must be updated and mamba_block_size must be reset from the sentinel.
    assert cache_config.mamba_cache_mode == "align"
    assert cache_config.mamba_block_size == 528, (
        "mamba_block_size must be reset from max_model_len sentinel to block_size")
    # mamba_page_size_padded was already set, so no second normalization pass.
    assert check_and_update_config.call_count == 1
    assert update_block_size_for_backend.call_count == 1
