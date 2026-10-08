"""CPU coverage for the proactive device->host Mamba spill (D1).

The spill fires at the mamba alloc choke point when the device allocator drops
below SGLANG_MAMBA_SPILL_LOW_WATER: the oldest safe device-only checkpoint is
D->H copied, and its VRAM slot is freed only at the write ack. Everything here
is selection/bookkeeping over fakes -- no controller, no GPU.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache import component_type as ct_mod
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

COMPS = (ComponentType.FULL, ComponentType.MAMBA)
MAMBA = ComponentType.MAMBA


def _node(parent, token: int, *, device: bool = True, host: bool = False):
    node = UnifiedTreeNode(COMPS)
    node.parent = parent
    node.key = RadixKey(array("q", range(token, token + 1)))
    if parent is not None:
        parent.children[node.key.child_key(1)] = node
    cd = node.component_data[MAMBA]
    cd.value = torch.tensor([token], dtype=torch.int64) if device else None
    cd.host_value = torch.tensor([1000 + token], dtype=torch.int64) if host else None
    node.component_data[ComponentType.FULL].value = [token]
    return node


class _FakeCore:
    """Minimal `self` running the REAL selection / spec methods unbound."""

    mamba_spill_oldest = UnifiedTreeCore.mamba_spill_oldest
    build_mamba_backup_spec = UnifiedTreeCore.build_mamba_backup_spec

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

    def add(self, parent, token, **kwargs):
        node = _node(parent, token, **kwargs)
        self._ids[node.id] = node
        if node.component_data[MAMBA].value is not None:
            self.lru_lists[MAMBA].insert_mru(node)
        return node

    def node_by_id(self, node_id):
        return self._ids[node_id]


class TestMambaSpillSelection(unittest.TestCase):
    def setUp(self):
        self._saved = (ct_mod._mamba_spill_low_water_cache, ct_mod._mamba_host_anchor_cache)

    def tearDown(self):
        (
            ct_mod._mamba_spill_low_water_cache,
            ct_mod._mamba_host_anchor_cache,
        ) = self._saved

    def test_oldest_device_only_node_selected_first(self):
        core = _FakeCore()
        oldest = core.add(core.root_node, 1)
        middle = core.add(oldest, 2)
        tip = core.add(middle, 3)

        self.assertEqual(core.mamba_spill_oldest(1), [oldest.id])
        self.assertEqual(
            core.mamba_spill_oldest(8), [oldest.id, middle.id, tip.id]
        )

    def test_locked_session_and_pending_nodes_are_skipped(self):
        core = _FakeCore()
        a = core.add(core.root_node, 1)
        b = core.add(a, 2)
        c = core.add(b, 3)
        d = core.add(c, 4)
        already = core.add(d, 5, device=True, host=True)

        b.component_data[MAMBA].lock_ref = 1
        c.component_data[MAMBA].session_ref = 2
        d.write_through_pending_id = d.id
        core.add(core.root_node, 6).load_back_pending_id = 7

        self.assertEqual(core.mamba_spill_oldest(8), [a.id])
        self.assertIsNotNone(already)  # host-backed: never a victim

    def test_node_absent_from_lru_is_not_selected(self):
        core = _FakeCore()
        a = core.add(core.root_node, 1)
        core.lru_lists[MAMBA].remove_node(a)
        self.assertEqual(core.mamba_spill_oldest(8), [])

    def test_spec_is_empty_for_a_host_backed_node(self):
        core = _FakeCore()
        node = core.add(core.root_node, 1, host=True)
        self.assertIsNone(core.build_mamba_backup_spec(node.id))


class TestMambaSpillTrigger(unittest.TestCase):
    def setUp(self):
        self._saved = (ct_mod._mamba_spill_low_water_cache, ct_mod._mamba_host_anchor_cache)

    def tearDown(self):
        (
            ct_mod._mamba_spill_low_water_cache,
            ct_mod._mamba_host_anchor_cache,
        ) = self._saved

    def _component(self, *, available=0):
        component = object.__new__(MambaComponent)
        component.tree_core = SimpleNamespace(enable_hicache=True)
        component.cache = SimpleNamespace(
            _component_available_size=lambda ct: available,
            buffer_pipeline=None,
            _spill_calls=[],
            _execute_mamba_backup=lambda nid: component.cache._spill_calls.append(nid)
            or True,
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(
                    alloc=lambda n: None, available_size=lambda: available
                )
            ),
            evict_for_alloc=lambda params: None,
        )
        return component

    def test_threshold_zero_is_a_no_op(self):
        ct_mod._mamba_spill_low_water_cache = 0
        ct_mod._mamba_host_anchor_cache = True
        component = self._component()
        component.tree_core.mamba_spill_oldest = lambda n: [1, 2]
        component._maybe_spill_low_water()
        self.assertEqual(component.cache._spill_calls, [])

    def test_anchor_off_is_a_no_op(self):
        ct_mod._mamba_spill_low_water_cache = 4
        ct_mod._mamba_host_anchor_cache = False
        component = self._component()
        component.tree_core.mamba_spill_oldest = lambda n: [1]
        component._maybe_spill_low_water()
        self.assertEqual(component.cache._spill_calls, [])

    def test_above_low_water_is_a_no_op(self):
        ct_mod._mamba_spill_low_water_cache = 4
        ct_mod._mamba_host_anchor_cache = True
        component = self._component(available=4)
        component.tree_core.mamba_spill_oldest = lambda n: [1]
        component._maybe_spill_low_water()
        self.assertEqual(component.cache._spill_calls, [])

    def test_below_low_water_spills_capped_at_eight(self):
        ct_mod._mamba_spill_low_water_cache = 4
        ct_mod._mamba_host_anchor_cache = True
        component = self._component(available=3)
        component.tree_core.mamba_spill_oldest = lambda n: [i for i in range(n)]
        component._maybe_spill_low_water()
        self.assertEqual(component.cache._spill_calls, list(range(8)))


class TestMambaSpillBackup(unittest.TestCase):
    """`_execute_mamba_backup` / `_finish_mamba_spill` over a faked controller."""

    def _cache(self, core, node):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = core
        cache.tree_components = COMPS
        cache.components = {MAMBA: SimpleNamespace(component_type=MAMBA)}
        cache.buffer_pipeline = None
        cache.metrics_collector = None
        cache.ongoing_write_through = {}
        cache.ongoing_mamba_spill = {}
        cache.ongoing_load_back = {}
        host = torch.arange(4, dtype=torch.int64)
        cache.host_pool_group = SimpleNamespace(
            alloc=lambda n, pool=None, reclaim=None: host[:n],
            free=lambda idx, pool=None: None,
        )
        cache.cache_controller = SimpleNamespace(
            write=lambda device, node_id=-1, extra_pools=None: torch.empty(
                0, dtype=torch.int64
            )
        )
        cache.evict_host = lambda n, ct: n
        cache.enable_storage = False
        cache._free_values = lambda d, h: None
        core.commit_backup = lambda nid, hv, xfers: node.component_data[
            MAMBA
        ].__setattr__("host_value", hv if hv.numel() else torch.tensor([7]))
        core.mark_write_through_pending = lambda nid: node.__setattr__(
            "write_through_pending_id", nid
        )
        def _tombstone(n, comp, df, hf, target=None, tracker=None):
            n.component_data[MAMBA].value = None
            return 1, 0

        core._evict_component_and_detach_lru = _tombstone
        core._for_each_component_lru = lambda *a, **k: None
        core._update_evictable_leaf_sets = lambda n: None
        core.finish_write_through = lambda nids, ack: node.__setattr__(
            "write_through_pending_id", None
        )
        return cache

    def test_backup_sets_host_value_and_registers_the_ack(self):
        core = _FakeCore()
        node = core.add(core.root_node, 1)
        cache = self._cache(core, node)

        self.assertTrue(cache._execute_mamba_backup(node.id))
        self.assertIsNotNone(node.component_data[MAMBA].host_value)
        self.assertIsNotNone(node.component_data[MAMBA].value)  # freed at ack only
        self.assertEqual(cache.ongoing_mamba_spill[node.id], node.id)

        cache._finish_write_through_ack(node.id)
        self.assertIsNone(node.component_data[MAMBA].value)
        self.assertEqual(cache.ongoing_mamba_spill, {})
        self.assertIsNone(node.write_through_pending_id)

    def test_same_node_is_never_spilled_twice(self):
        core = _FakeCore()
        node = core.add(core.root_node, 1)
        cache = self._cache(core, node)
        self.assertTrue(cache._execute_mamba_backup(node.id))
        # host_value now set and the pending mark raised: both guards reject.
        self.assertFalse(cache._execute_mamba_backup(node.id))
        self.assertEqual(core.mamba_spill_oldest(8), [])
        self.assertEqual(len(cache.ongoing_write_through), 1)

    def test_failed_controller_write_frees_the_host_slot(self):
        core = _FakeCore()
        node = core.add(core.root_node, 1)
        cache = self._cache(core, node)
        freed = []
        cache.host_pool_group.free = lambda idx, pool=None: freed.append(idx)
        cache.cache_controller.write = lambda **kw: None
        cache.cache_controller.write = (
            lambda device, node_id=-1, extra_pools=None: None
        )
        self.assertFalse(cache._execute_mamba_backup(node.id))
        self.assertEqual(len(freed), 1)
        self.assertEqual(cache.ongoing_write_through, {})


if __name__ == "__main__":
    unittest.main()
