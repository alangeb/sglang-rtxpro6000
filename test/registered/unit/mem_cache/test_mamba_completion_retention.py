"""CPU coverage for completion-boundary Mamba checkpoint retention (D2).

When SGLANG_MAMBA_COMPLETION_RETENTION=1, the terminal Mamba state of a finished
completion is marked as a completion boundary, and the proactive D1 spill
prefers demoting completion-boundary checkpoints to host over older non-boundary
checkpoints under device mamba pool pressure. Everything here is selection over
fakes -- no controller, no GPU. The flag-off path must be byte-identical to the
pre-D2 single walk.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache import component_type as ct_mod
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

COMPS = (ComponentType.FULL, ComponentType.MAMBA)
MAMBA = ComponentType.MAMBA


def _node(parent, token: int, *, device: bool = True, host: bool = False,
          full_host: bool = True, completion_boundary: bool = False):
    node = UnifiedTreeNode(COMPS)
    node.parent = parent
    node.key = RadixKey(array("q", range(token, token + 1)))
    if parent is not None:
        parent.children[node.key.child_key(1)] = node
    cd = node.component_data[MAMBA]
    cd.value = torch.tensor([token], dtype=torch.int64) if device else None
    cd.host_value = torch.tensor([1000 + token], dtype=torch.int64) if host else None
    node.component_data[ComponentType.FULL].value = [token]
    node.component_data[ComponentType.FULL].host_value = (
        torch.tensor([2000 + token], dtype=torch.int64) if full_host else None
    )
    node.completion_boundary = completion_boundary
    return node


class _FakeCore:
    """Minimal `self` running the REAL selection methods unbound."""

    mamba_spill_oldest = UnifiedTreeCore.mamba_spill_oldest
    _mamba_spill_victim_eligible = UnifiedTreeCore._mamba_spill_victim_eligible

    def __init__(self):
        self.root_node = _node(None, 0, device=False)
        self.lru_lists = {MAMBA: UnifiedLRUList(MAMBA, COMPS)}
        self.host_lru_lists = {MAMBA: UnifiedLRUList(MAMBA, COMPS, use_host_ptr=True)}
        self.components_by_type = {
            MAMBA: SimpleNamespace(
                session_ref=lambda n: n.component_data[MAMBA].session_ref,
                build_hicache_transfers=lambda n, p: (
                    [
                        PoolTransfer(
                            name=PoolName.MAMBA,
                            device_indices=n.component_data[MAMBA].value,
                        )
                    ]
                    if n.component_data[MAMBA].value is not None
                    else None
                ),
            )
        }
        self.enable_hicache = True
        self._ids = {}
        self._completion_boundary_dropped = 0

    def add(self, parent, token, **kwargs):
        node = _node(parent, token, **kwargs)
        self._ids[node.id] = node
        if node.component_data[MAMBA].value is not None:
            self.lru_lists[MAMBA].insert_mru(node)
        return node

    def node_by_id(self, node_id):
        return self._ids[node_id]


class TestCompletionBoundaryRetention(unittest.TestCase):
    def setUp(self):
        self._saved = ct_mod._mamba_completion_retention_cache

    def tearDown(self):
        ct_mod._mamba_completion_retention_cache = self._saved

    def test_flag_off_is_oldest_first_ignoring_boundaries(self):
        """Flag off: byte-identical to the pre-D2 single walk (oldest first,
        completion_boundary ignored)."""
        ct_mod._mamba_completion_retention_cache = False
        core = _FakeCore()
        oldest = core.add(core.root_node, 1, completion_boundary=True)
        middle = core.add(oldest, 2, completion_boundary=False)
        tip = core.add(middle, 3, completion_boundary=False)
        # Oldest first, even though the oldest is a completion boundary.
        self.assertEqual(core.mamba_spill_oldest(1), [oldest.id])
        self.assertEqual(core.mamba_spill_oldest(8), [oldest.id, middle.id, tip.id])
        self.assertEqual(core._completion_boundary_dropped, 0)

    def test_flag_on_prefers_completion_boundaries(self):
        """Flag on: a NEWER completion boundary is preferred over an OLDER
        non-boundary checkpoint (the most-valuable rewind target is demoted
        first)."""
        ct_mod._mamba_completion_retention_cache = True
        core = _FakeCore()
        oldest = core.add(core.root_node, 1, completion_boundary=False)
        middle = core.add(oldest, 2, completion_boundary=False)
        tip = core.add(middle, 3, completion_boundary=True)
        # tip (newer, boundary) comes before oldest/middle (older, non-boundary).
        self.assertEqual(core.mamba_spill_oldest(1), [tip.id])
        self.assertEqual(core.mamba_spill_oldest(8), [tip.id, oldest.id, middle.id])
        self.assertEqual(core._completion_boundary_dropped, 0)

    def test_flag_on_multiple_boundaries_keep_oldest_first(self):
        """Flag on: among completion boundaries, oldest-first order is kept."""
        ct_mod._mamba_completion_retention_cache = True
        core = _FakeCore()
        b1 = core.add(core.root_node, 1, completion_boundary=True)
        b2 = core.add(b1, 2, completion_boundary=True)
        non = core.add(b2, 3, completion_boundary=False)
        self.assertEqual(core.mamba_spill_oldest(8), [b1.id, b2.id, non.id])

    def test_flag_on_boundary_without_full_host_is_dropped_not_victim(self):
        """A completion boundary that is device-only without a FULL host copy
        cannot be demoted to host (tree invariant); it is counted as at-risk
        (dropped) and never selected as a victim."""
        ct_mod._mamba_completion_retention_cache = True
        core = _FakeCore()
        at_risk = core.add(core.root_node, 1, completion_boundary=True,
                           full_host=False)
        safe = core.add(at_risk, 2, completion_boundary=False, full_host=True)
        # at_risk is NOT a victim (no FULL host copy); safe is.
        self.assertEqual(core.mamba_spill_oldest(8), [safe.id])
        self.assertEqual(core._completion_boundary_dropped, 1)

    def test_flag_on_host_backed_boundary_never_a_victim(self):
        """A completion boundary that already has a host copy is never a spill
        victim (nothing to demote); it is not counted as dropped either."""
        ct_mod._mamba_completion_retention_cache = True
        core = _FakeCore()
        host_backed = core.add(core.root_node, 1, completion_boundary=True,
                               host=True)
        other = core.add(host_backed, 2, completion_boundary=False)
        self.assertEqual(core.mamba_spill_oldest(8), [other.id])
        self.assertEqual(core._completion_boundary_dropped, 0)


if __name__ == "__main__":
    unittest.main()
