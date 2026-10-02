# SPDX-License-Identifier: Apache-2.0
"""Host cgroup headroom check shared by the EngineCore sleep guard and the engine reconfigure hook.

Kept free of HPU imports so the engine process can use it without loading the device stack.
"""
import os
from typing import Any

SKIP_HOST_GUARD_ENV = "VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD"


def host_guard_disabled() -> bool:
    return os.environ.get(SKIP_HOST_GUARD_ENV, "0") == "1"


def raise_if_reports_exceed_host_headroom(worker_reports: Any, *, action: str, outcome: str) -> None:
    """Raise when the workers' device-resident model bytes would not fit in host memory.

    Each report is ``{"host", "required_bytes", "headroom_bytes"}`` from
    ``HPUWorker.check_sleep_host_headroom``. Workers on one host share a cgroup and move
    their shards to CPU concurrently, so requirements are summed per host and compared
    with the smallest headroom reported on that host. Missing or malformed reports pass.
    """
    if not isinstance(worker_reports, (list, tuple)):
        return
    per_host: dict[Any, tuple[int, int | None]] = {}
    for report in worker_reports:
        if not isinstance(report, dict):
            continue
        host = report.get("host")
        required, headroom = per_host.get(host, (0, None))
        required += int(report.get("required_bytes") or 0)
        reported = report.get("headroom_bytes")
        if isinstance(reported, int):
            headroom = reported if headroom is None else min(headroom, reported)
        per_host[host] = (required, headroom)
    for required, headroom in per_host.values():
        if required > 0 and headroom is not None and headroom < required:
            raise RuntimeError(f"{action} aborted: insufficient host memory to move the model to CPU "
                               f"(required={required / 2**30:.1f}GiB, cgroup_headroom={headroom / 2**30:.1f}GiB); "
                               f"{outcome}. Set {SKIP_HOST_GUARD_ENV}=1 to bypass.")
