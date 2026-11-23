"""Collect one complete actor PPO minibatch using verl's NPU profiler.

The outer DistProfiler supplies the rank/global-step gates. This module only
activates inside actor_update and keeps load/offload and later minibatches out.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


_SESSION = ContextVar("actor_minibatch_profile_session", default=None)


@contextmanager
def actor_update_scope(backend):
    token = _SESSION.set({"backend": backend, "capturing": False, "phases": {}})
    try:
        yield
    finally:
        _SESSION.reset(token)


@contextmanager
def profile_actor_minibatch(batch_idx):
    session = _SESSION.get()
    if session is None:
        yield
        return
    backend = session["backend"]
    if batch_idx != 0 or backend._actor_minibatch_captured:
        yield
        return

    from .mstx_profile import get_npu_profiler, mark_end_range, mark_start_range, torch_npu

    # Mark consumed before entering training: errors must never re-run an update.
    backend._actor_minibatch_captured = True
    step = backend._profile_step
    if step is None:
        raise RuntimeError("Actor minibatch profiling requires an explicit profile_step")
    output = (
        Path(backend.profile_save_path)
        / f"step_{int(step):06d}"
        / "actor_ppo_minibatch"
        / f"rank_{backend.rank:05d}"
    )
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "global_step": int(step),
        "rank": backend.rank,
        "ppo_minibatch_index": batch_idx,
        "completed": False,
        "scope": "minibatch preparation, all microbatch forward/backward, optimizer, and metrics",
    }
    prof = get_npu_profiler(
        contents=backend.profile_contents,
        profile_level=backend.profile_level,
        profile_save_path=str(output),
        analysis=backend.analysis,
    )
    torch_npu.npu.synchronize()
    prof.start()
    session["capturing"] = True
    marker = None
    try:
        marker = mark_start_range(f"actor_ppo_minibatch_{batch_idx}")
        yield
        report["completed"] = True
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        try:
            torch_npu.npu.synchronize()
            if marker is not None:
                mark_end_range(marker)
        except BaseException as exc:
            report["completed"] = False
            report["finalize_error_type"] = type(exc).__name__
            raise
        finally:
            session["capturing"] = False
            try:
                prof.stop()
            except BaseException:
                report["completed"] = False
                report["export_failed"] = True
                raise
            finally:
                report["phase_counts"] = session["phases"]
                temporary = output / "capture.json.tmp"
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(output / "capture.json")
        if report["completed"]:
            print(f"ACTOR_PPO_MINIBATCH_PROFILE_COMPLETE: {output}", flush=True)


@contextmanager
def profile_phase(name):
    session = _SESSION.get()
    if session is None or not session["capturing"]:
        yield
        return
    from .mstx_profile import mark_end_range, mark_start_range

    category = name.split("/", 1)[0]
    session["phases"][category] = session["phases"].get(category, 0) + 1
    marker = mark_start_range(name)
    try:
        yield
    finally:
        mark_end_range(marker)
