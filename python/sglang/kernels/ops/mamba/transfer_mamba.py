"""JIT-compiled Mamba KV cache transfer kernel.

Provides ``transfer_kv_mamba_pf_lf`` (load: page_first -> layer_first)
and ``transfer_kv_mamba_lf_pf`` (backup: layer_first -> page_first).

Uses the shared ``load_jit`` + ``cache_once`` infrastructure from
``sglang.kernels.jit.utils`` — the same mechanism used by ``hicache.py``
for MHA/MLA staged write-back kernels.  This ensures consistent
content-addressed caching, CUDA arch detection, and multi-worker
JIT compilation behavior across all JIT kernels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _jit_transfer_mamba_module() -> Module:
    return load_jit(
        "transfer_mamba",
        cuda_files=["kvcacheio/transfer_mamba.cuh"],
        cuda_wrappers=[
            ("transfer_kv_mamba_pf_lf", "&TransferMambaKernel::run_pf_lf"),
            ("transfer_kv_mamba_lf_pf", "&TransferMambaKernel::run_lf_pf"),
        ],
    )


_rejected_mamba_indices = {"n": 0}
_guard_checked = {"n": 0}


def _guard_indices(name, idx, bound):
    """P2-D6: reject out-of-range / negative mamba slot ids BEFORE the copy.
    The kernels dereference src_indices/dst_indices with no bounds check
    (transfer_mamba.cuh:36-41, :66-71), so a stale or int8-mapped id is an
    immediate illegal-memory-access. Gated: with the flag off this is a no-op."""
    from sglang.srt.mem_cache.unified_cache.component_type import mamba_host_anchor_enabled
    if not mamba_host_anchor_enabled() or idx is None or idx.numel() == 0:
        return idx
    bad = (idx < 0) | (idx >= bound)
    _guard_checked["n"] += 1
    if _guard_checked["n"] <= 3:
        import logging

        logging.getLogger(__name__).info(
            "[MAMBA-HOST] D6 guard ACTIVE on %s numel=%d bound=%d n=%d",
            name, int(idx.numel()), int(bound), _guard_checked["n"],
        )
    if bool(bad.any().item()):
        _rejected_mamba_indices["n"] += int(bad.sum().item())
        raise ValueError(
            f"[MAMBA-HOST] D6: {name} index out of range (bound={bound}) "
            f"min={int(idx.min())} max={int(idx.max())} "
            f"rejected_total={_rejected_mamba_indices['n']}"
        )
    return idx


@debug_kernel_api
def transfer_kv_mamba_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    num_warps_per_item: int = 32,
):
    _guard_indices("pf_lf.src", src_indices, src.size(0))
    _guard_indices("pf_lf.dst", dst_indices, dst.size(0))
    module = _jit_transfer_mamba_module()
    module.transfer_kv_mamba_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
    )


@debug_kernel_api
def transfer_kv_mamba_lf_pf(
    src_ptrs: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    num_warps_per_item: int = 32,
):
    _guard_indices("lf_pf.src", src_indices, dst.size(0))
    _guard_indices("lf_pf.dst", dst_indices, dst.size(0))
    module = _jit_transfer_mamba_module()
    module.transfer_kv_mamba_lf_pf(
        src_ptrs,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
    )
