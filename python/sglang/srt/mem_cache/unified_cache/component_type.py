"""The per-attention component identity; a leaf module importable from anywhere."""

import os
from enum import Enum
from typing import Any, Optional


class ComponentType(int, Enum):
    """Integer enum so that per-node list/tuple storage can be indexed directly."""

    FULL = 0
    SWA = 1
    MAMBA = 2
    C128 = 3

    def __str__(self) -> str:  # keep human-readable logging
        return self.name.lower()

    @property
    def is_full(self) -> bool:
        return self == ComponentType.FULL

    @property
    def is_swa(self) -> bool:
        return self == ComponentType.SWA

    @property
    def is_mamba(self) -> bool:
        return self == ComponentType.MAMBA


BASE_COMPONENT_TYPE = ComponentType.FULL


# ---------------------------------------------------------------------------
# P1 (SGLANG_MAMBA_HOST_ANCHOR): widen the host-tier anchor to Mamba.
#
# The tree core decides host-tier survival from ComponentType.FULL only
# (unified_tree_core.py:140-151 `backuped`/`evicted`, :1694-1708
# `_is_host_leaf`, :1622-1636 tombstone walk). For a hybrid/mamba model the
# Mamba state is a *whole-slot* copy that is worth more than the KV slice it
# rides with, so losing it on a FULL-only test is what makes the host tier a
# silent no-op (upstream issue #33713). `host_anchored()` is the single widened
# test used by the few sites that gate survival. Default OFF: with the env flag
# unset every call returns False and the code path is byte-identical to the
# untouched tree, so A/B runs share one binary.
# ---------------------------------------------------------------------------

MAMBA_HOST_ANCHOR_ENV = "SGLANG_MAMBA_HOST_ANCHOR"
_mamba_host_anchor_cache: Optional[bool] = None
_mamba_host_debug_cache: Optional[bool] = None


def mamba_host_anchor_enabled() -> bool:
    """Cached read of SGLANG_MAMBA_HOST_ANCHOR (default off)."""
    global _mamba_host_anchor_cache
    if _mamba_host_anchor_cache is None:
        _mamba_host_anchor_cache = (
            os.environ.get(MAMBA_HOST_ANCHOR_ENV, "0").strip() == "1"
        )
    return _mamba_host_anchor_cache


def mamba_host_debug_enabled() -> bool:
    """Separate observability flag: logs the producer + the aux-only load-back.
    Independent of SGLANG_MAMBA_HOST_ANCHOR so the control arm can be observed too."""
    global _mamba_host_debug_cache
    if _mamba_host_debug_cache is None:
        _mamba_host_debug_cache = (
            os.environ.get("SGLANG_MAMBA_HOST_DEBUG", "0").strip() == "1"
        )
    return _mamba_host_debug_cache


_mamba_spill_low_water_cache: Optional[int] = None


def mamba_spill_low_water() -> int:
    """High-water trigger for the PROACTIVE mamba spill (D1).

    When the device mamba allocator drops below this many free slots, the
    oldest safe device-only checkpoint is D->H copied so its VRAM slot can be
    reclaimed by normal eviction, instead of waiting for an allocation to fail
    (which never happens: tree.evict() can always free one slot by destroying
    the oldest checkpoint -- but destroys it *without* a host copy).

    0 disables the proactive trigger, and 0 is the DEFAULT (was 4 while the
    function had no call sites). It is now wired into the mamba alloc choke
    point, so a non-zero default would silently change behaviour for every
    host-anchor run that never opted in. With the env unset, behaviour is
    identical to the pre-D1 tree; the spill additionally requires
    SGLANG_MAMBA_HOST_ANCHOR=1 and an enabled hierarchical cache.
    """
    global _mamba_spill_low_water_cache
    if _mamba_spill_low_water_cache is None:
        try:
            _mamba_spill_low_water_cache = max(
                0, int(os.environ.get("SGLANG_MAMBA_SPILL_LOW_WATER", "0"))
            )
        except ValueError:
            _mamba_spill_low_water_cache = 0
    return _mamba_spill_low_water_cache


MAMBA_ONLY_HOST_CACHE_ENV = "SGLANG_MAMBA_ONLY_HOST_CACHE"
_mamba_only_host_cache_cache: Optional[bool] = None


def mamba_only_host_cache_enabled() -> bool:
    """Cached read of SGLANG_MAMBA_ONLY_HOST_CACHE (default off).

    Lets the host tier exist for Mamba checkpoints alone. Without it the
    assembler builds host pools only when a hicache budget exists, so
    SGLANG_MAMBA_HOST_SIZE_GB would have nowhere to live with --hicache-size 0.
    With this flag on *and* an explicit Mamba host size, the host stack is built
    with a minimal KV/QSA placeholder pool. True zero-KV-host is not
    architecturally possible: FULL is the tree anchor that every backup /
    load-back transfer rides on, and sanity_check PART 2 requires aux host data
    to be backed by Full host data. Default OFF: unset is upstream behaviour.
    """
    global _mamba_only_host_cache_cache
    if _mamba_only_host_cache_cache is None:
        _mamba_only_host_cache_cache = (
            os.environ.get(MAMBA_ONLY_HOST_CACHE_ENV, "0").strip() == "1"
        )
    return _mamba_only_host_cache_cache


MAMBA_COMPLETION_RETENTION_ENV = "SGLANG_MAMBA_COMPLETION_RETENTION"
_mamba_completion_retention_cache: Optional[bool] = None


def mamba_completion_retention_enabled() -> bool:
    """Cached read of SGLANG_MAMBA_COMPLETION_RETENTION (default off).

    D2: completion-boundary Mamba checkpoint retention. When on, the terminal
    Mamba state of a finished OpenAI/chat completion is marked as a completion
    boundary, and the proactive D1 spill prefers demoting completion-boundary
    checkpoints to host (preserving them) over older non-boundary checkpoints
    when device mamba pool pressure fires. Default OFF: with the env unset the
    behaviour is byte-identical to the pre-D2 tree.
    """
    global _mamba_completion_retention_cache
    if _mamba_completion_retention_cache is None:
        _mamba_completion_retention_cache = (
            os.environ.get(MAMBA_COMPLETION_RETENTION_ENV, "0").strip() == "1"
        )
    return _mamba_completion_retention_cache


def host_anchored(node: Any) -> bool:
    """True if *node* may stay in the tree on host-tier evidence alone.

    Satisfied by a FULL host copy (the upstream test) or, when the flag is on,
    by a Mamba host copy. Used only where the current code *requires* FULL to
    keep a node alive -- never to claim a node is fully restorable.
    """
    if not mamba_host_anchor_enabled():
        return False
    if node.component_data[ComponentType.FULL].host_value is not None:
        return True
    if ComponentType.MAMBA not in node.component_types:
        return False
    return node.component_data[ComponentType.MAMBA].host_value is not None
