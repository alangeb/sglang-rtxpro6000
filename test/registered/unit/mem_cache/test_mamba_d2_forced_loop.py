"""CPU coverage that FORCES the D2 completion-boundary loop end to end.

The production trial proved the general mamba host tier restores (20 host hits,
0 crashes) but left the two D2 payoff counters at zero: `demoted` (a boundary
chosen as the spill victim) and `restored` (a rewind consuming a host-only
boundary). Those only fire under device mamba pressure plus a rewind, which is
hard to force from the outside. This test drives the real methods over fakes so
the whole capture -> preferential demote -> restore chain is exercised and
counted on CPU, with no controller and no GPU.

Chain exercised:
  mamba_spill_oldest        (D2 two-pass selection: boundaries first)
  _spill_oldest_to_host     (issues backups, counts demoted/dropped)
  prepare_load_back         (counts restored on a host-only boundary)
  collect_full_device_indices (PER_SLOT ancestor guard -- the crash fix)
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache import component_type as ct_mod
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
    PrepareLoadBackResult,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

COMPS = (ComponentType.FULL, ComponentType.MAMBA)
MAMBA = ComponentType.MAMBA
FULL = ComponentType.FULL


class _CountingCollector:
    """Stands in for the metrics collector: records D2 counter increments."""

    def __init__(self):
        self.captured = 0
        self.demoted = 0
        self.restored = 0
        self.dropped = 0

    def increment_mamba_completion_boundary_captured(self, num=1):
        self.captured += num

    def increment_mamba_completion_boundary_demoted(self, num=1):
        self.demoted += num

    def increment_mamba_completion_boundary_restored(self, num=1):
        self.restored += num

    def increment_mamba_completion_boundary_dropped(self, num=1):
        self.dropped += num


def _node(parent, token, *, device=True, host=False, full_host=True,
          completion_boundary=False):
    node = UnifiedTreeNode(COMPS)
    node.parent = parent
    node.key = RadixKey(array("q", range(token, token + 1)))
    if parent is not None:
        parent.children[node.key.child_key(1)] = node
    cd = node.component_data[MAMBA]
    cd.value = torch.tensor([token], dtype=torch.int64) if device else None
    cd.host_value = torch.tensor([1000 + token], dtype=torch.int64) if host else None
    node.component_data[FULL].value = (
        torch.tensor([token], dtype=torch.int64) if device else None
    )
    node.component_data[FULL].host_value = (
        torch.tensor([2000 + token], dtype=torch.int64) if full_host else None
    )
    node.completion_boundary = completion_boundary
    return node


class _FakeCore:
    """Minimal `self` running the REAL selection/guard methods unbound."""

    mamba_spill_oldest = UnifiedTreeCore.mamba_spill_oldest
    _mamba_spill_victim_eligible = UnifiedTreeCore._mamba_spill_victim_eligible
    collect_full_device_indices = UnifiedTreeCore.collect_full_device_indices
    component_has_host_value_only = UnifiedTreeCore.component_has_host_value_only

    def __init__(self):
        self.root_node = _node(None, 0, device=False)
        # Mirror production: the root carries an empty (not None) FULL value, so
        # a runaway walk passes the assert and dies on root.parent is None.
        self.root_node.component_data[FULL].value = []
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
        self._ids = {self.root_node.id: self.root_node}
        self._empty_match_result = SimpleNamespace(
            device_indices=torch.empty(0, dtype=torch.int64)
        )
        self._completion_boundary_dropped = 0

    def add(self, parent, token, **kwargs):
        node = _node(parent, token, **kwargs)
        self._ids[node.id] = node
        if node.component_data[MAMBA].value is not None:
            self.lru_lists[MAMBA].insert_mru(node)
        return node

    def node_by_id(self, node_id):
        return self._ids[node_id]


def _component(core, collector):
    """MambaComponent wired to the fake core, with backup/alloc stubs."""
    component = object.__new__(MambaComponent)
    component.component_type = MAMBA
    component.tree_core = core
    spilled = []

    def _backup(node_id):
        node = core.node_by_id(node_id)
        node.component_data[MAMBA].host_value = node.component_data[MAMBA].value
        node.component_data[MAMBA].value = None  # device slot freed at ack
        core.host_lru_lists[MAMBA].insert_mru(node)
        spilled.append(node_id)
        return True

    component.cache = SimpleNamespace(
        metrics_collector=collector,
        _execute_mamba_backup=_backup,
        req_to_token_pool=SimpleNamespace(
            mamba_allocator=SimpleNamespace(
                alloc=lambda n: torch.tensor([77], dtype=torch.int64),
                available_size=lambda: 0,
            )
        ),
        evict_for_alloc=lambda params: None,
    )
    component._spilled = spilled
    return component


class _FakeReq:
    def __init__(self):
        self.mamba_pool_idx = None


class TestD2ForcedEndToEnd(unittest.TestCase):
    """Force the full D2 loop: capture, preferential demote, rewind restore."""

    def setUp(self):
        self._saved = (
            ct_mod._mamba_completion_retention_cache,
            ct_mod._mamba_host_anchor_cache,
        )
        ct_mod._mamba_completion_retention_cache = True
        ct_mod._mamba_host_anchor_cache = True

    def tearDown(self):
        (
            ct_mod._mamba_completion_retention_cache,
            ct_mod._mamba_host_anchor_cache,
        ) = self._saved

    def test_demote_then_restore_counters_move(self):
        core = _FakeCore()
        collector = _CountingCollector()
        # Oldest non-boundary, then a NEWER completion boundary at the tip.
        plain = core.add(core.root_node, 1, completion_boundary=False)
        tip = core.add(plain, 2, completion_boundary=True)
        component = _component(core, collector)

        # Device pressure: spill. D2 must pick the boundary FIRST (demoted) even
        # though it is newer than the non-boundary.
        spilled = component._spill_oldest_to_host(1)
        self.assertEqual(spilled, 1)
        self.assertEqual(component._spilled, [tip.id])
        self.assertEqual(collector.demoted, 1, "boundary must be counted demoted")
        self.assertEqual(collector.dropped, 0)
        self.assertIsNone(tip.component_data[MAMBA].value, "device slot freed")
        self.assertIsNotNone(tip.component_data[MAMBA].host_value, "host copy kept")

        # Rewind onto the now host-only boundary: prepare_load_back must count a
        # restore and allocate a device slot.
        req = _FakeReq()
        result = component.prepare_load_back(tip.id, req=req)
        self.assertIsInstance(result, PrepareLoadBackResult)
        self.assertIsNotNone(result.allocated_mamba_slot)
        self.assertEqual(int(req.mamba_pool_idx), 77)
        self.assertEqual(collector.restored, 1, "boundary rewind must count restored")

    def test_restore_not_counted_on_non_boundary(self):
        """A host-only node that is NOT a completion boundary loads back but must
        not inflate the D2 restored counter."""
        core = _FakeCore()
        collector = _CountingCollector()
        plain = core.add(
            core.root_node, 1, completion_boundary=False, device=False, host=True
        )
        component = _component(core, collector)
        req = _FakeReq()
        component.prepare_load_back(plain.id, req=req)
        self.assertEqual(int(req.mamba_pool_idx), 77, "load-back still happens")
        self.assertEqual(collector.restored, 0, "non-boundary must not count")

    def test_restore_inert_when_retention_flag_off(self):
        """Flag off: the load-back happens but D2 stays inert (no counter)."""
        ct_mod._mamba_completion_retention_cache = False
        core = _FakeCore()
        collector = _CountingCollector()
        tip = core.add(
            core.root_node, 1, completion_boundary=True, device=False, host=True
        )
        component = _component(core, collector)
        req = _FakeReq()
        component.prepare_load_back(tip.id, req=req)
        self.assertEqual(int(req.mamba_pool_idx), 77)
        self.assertEqual(collector.restored, 0, "D2 inert when flag off")

    def test_dropped_boundary_counted_not_demoted(self):
        """A device-only boundary with no FULL host copy cannot be demoted; it is
        counted at-risk (dropped) and never spilled."""
        core = _FakeCore()
        collector = _CountingCollector()
        at_risk = core.add(
            core.root_node, 1, completion_boundary=True, full_host=False
        )
        safe = core.add(at_risk, 2, completion_boundary=False, full_host=True)
        component = _component(core, collector)
        component._spill_oldest_to_host(8)
        self.assertEqual(component._spilled, [safe.id], "at-risk never spilled")
        self.assertEqual(collector.demoted, 0)
        self.assertEqual(collector.dropped, 1, "at-risk boundary counted dropped")


class TestPerSlotAncestorGuard(unittest.TestCase):
    """Regression for the :8001 crash: 'NoneType' object has no attribute
    'component_data' in collect_full_device_indices.

    Under SGLANG_MAMBA_PER_SLOT the device anchor advances on FULL alone, so
    until_node (last_node) can be DEEPER than from_node (best_match_node). The
    ancestor walk then climbs past from_node toward the root and off the top of
    the tree into None. The guard must return empty instead of dereferencing."""

    def setUp(self):
        self.core = _FakeCore()

    def test_walk_off_top_returns_empty(self):
        # root <- a <- b ; call with from=a, until=b (b is DEEPER, not an
        # ancestor of a). Walking up from a reaches root without hitting b.
        a = self.core.add(self.core.root_node, 1)
        b = self.core.add(a, 2)
        out = self.core.collect_full_device_indices(a.id, b.id)
        self.assertEqual(out.numel(), 0, "guard must return empty, not crash")

    def test_normal_direction_still_collects(self):
        """Sanity: when until_node IS a true ancestor, the walk still collects
        both nodes (the guard must not disable the normal path)."""
        a = self.core.add(self.core.root_node, 1)
        b = self.core.add(a, 2)
        out = self.core.collect_full_device_indices(
            b.id, self.core.root_node.id
        )
        self.assertGreaterEqual(out.numel(), 1, "normal walk collects")

    def test_root_as_from_node_returns_empty(self):
        """from == until == root: loop does not run -> empty result, no deref."""
        out = self.core.collect_full_device_indices(
            self.core.root_node.id, self.core.root_node.id
        )
        self.assertEqual(out.numel(), 0)


if __name__ == "__main__":
    unittest.main()
