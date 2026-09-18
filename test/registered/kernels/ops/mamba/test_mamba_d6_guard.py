"""Regression tests for the D6 bounds guards in the Mamba transfer kernel.

The backup-path guard (``transfer_kv_mamba_lf_pf``) bounds ``src_indices``
(DEVICE mamba slot ids) against the device-side pool size the caller passes
as ``src_bound``.  The historical fallback to ``dst.size(0)`` (the HOST pool
size) rejected legal device slots whenever the host tier is smaller than the
device tier (e.g. device 16 slots vs host 10 slots): the exact
``ValueError: [MAMBA-HOST] D6: lf_pf.src index out of range (bound=10)
min=13 max=13`` crash the q38-flash-next container hit on every start.

CPU-only: ``_guard_indices`` is pure Python over the index tensors.
"""

import inspect

import pytest
import torch

from sglang.kernels.ops.mamba import transfer_mamba
from sglang.srt.mem_cache.unified_cache import component_type
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE_SLOTS = 16
HOST_SLOTS = 10


def _set_anchor_flag(monkeypatch, value):
    # The env read is cached in a module global; set it directly so the test
    # is deterministic regardless of the process environment.
    monkeypatch.setattr(component_type, "_mamba_host_anchor_cache", value)


def test_lf_pf_src_accepts_legal_device_slot_beyond_host_size(monkeypatch):
    _set_anchor_flag(monkeypatch, True)
    # Device slots 10..15 are legal (device pool = 16) even though the host
    # tier only has 10 slots. Must NOT raise.
    src = torch.tensor([HOST_SLOTS, HOST_SLOTS + 3], dtype=torch.int64)
    transfer_mamba._guard_indices("lf_pf.src", src, DEVICE_SLOTS)


def test_lf_pf_src_rejects_out_of_range(monkeypatch):
    _set_anchor_flag(monkeypatch, True)
    src = torch.tensor([DEVICE_SLOTS], dtype=torch.int64)  # 16 is invalid
    with pytest.raises(ValueError, match="D6: lf_pf.src"):
        transfer_mamba._guard_indices("lf_pf.src", src, DEVICE_SLOTS)


def test_lf_pf_src_rejects_negative(monkeypatch):
    _set_anchor_flag(monkeypatch, True)
    src = torch.tensor([-1], dtype=torch.int64)
    with pytest.raises(ValueError, match="D6: lf_pf.src"):
        transfer_mamba._guard_indices("lf_pf.src", src, DEVICE_SLOTS)


def test_lf_pf_dst_still_bounded_by_host_size(monkeypatch):
    _set_anchor_flag(monkeypatch, True)
    # The dst (host) side keeps its own bound: host slot 10 is invalid.
    dst = torch.tensor([HOST_SLOTS], dtype=torch.int64)
    with pytest.raises(ValueError, match="D6: lf_pf.dst"):
        transfer_mamba._guard_indices("lf_pf.dst", dst, HOST_SLOTS)


def test_lf_pf_signature_exposes_src_bound_default_none():
    sig = inspect.signature(transfer_mamba.transfer_kv_mamba_lf_pf)
    assert "src_bound" in sig.parameters
    assert sig.parameters["src_bound"].default is None


def test_guard_noop_when_anchor_flag_off(monkeypatch):
    _set_anchor_flag(monkeypatch, False)
    # Default-off path is byte-identical to upstream: no raise even for
    # absurd ids, because the guard returns before checking.
    transfer_mamba._guard_indices(
        "lf_pf.src", torch.tensor([9999], dtype=torch.int64), HOST_SLOTS
    )
