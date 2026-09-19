"""Regression coverage for the Mamba host-anchor tombstone-cascade crash.

Production crash 2026-09-19 (q38-flash-next-pennyroyal): the match walk
widened for SGLANG_MAMBA_HOST_ANCHOR may stand on a Mamba-host-only bridge
node, but the tombstone cascade decided survival from FULL alone and deleted
that bridge, severing the ancestor chain under init_load_back ->
collect_full_device_indices walked off the root (AttributeError on None).

Fix under test: the cascade now keeps host_anchored() nodes alive (mirrors the
match guard), and collect_full_device_indices degrades to an empty restore
instead of crashing. Both no-op when the flag is off (default = upstream).
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache import component_type as ct_mod
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

COMPS = (ComponentType.FULL, ComponentType.MAMBA)


def _node(parent, token: int, *, full=None, mamba=None):
    """full/mamba: 'device' | 'host' | None (absent)."""
    node = UnifiedTreeNode(COMPS)
    node.parent = parent
    node.key = RadixKey(array("q", range(token, token + 1)))
    if parent is not None:
        parent.children[node.key.child_key(1)] = node
    for comp_type, residence in ((ComponentType.FULL, full), (ComponentType.MAMBA, mamba)):
        cd = node.component_data[comp_type]
        value = torch.tensor([token], dtype=torch.int64)
        if residence == "device":
            cd.value = value
        elif residence == "host":
            cd.host_value = value
    return node


class _FakeCore:
    """Minimal `self` for the two UnifiedTreeCore methods under test."""

    def __init__(self):
        self.root_node = _node(None, 0, full="device")
        self.page_size = 1
        self.components = []
        self.components_by_type = {}
        self.evictable_host_leaves = set()
        self.full_host_duplicates = {}
        self._empty_match_result = SimpleNamespace(
            device_indices=torch.tensor([], dtype=torch.int64)
        )
        self._ids = {}

    def _register(self, node):
        self._ids[node.id] = node

    def node_by_id(self, node_id):
        return self._ids[node_id]

    def _evict_component_and_detach_lru(self, *args, **kwargs):
        pass

    def _update_evictable_leaf_sets(self, node):
        pass

    def _remove_leaf_from_parent(self, node):
        # Mirrors the real method: unlink from children only; the cascade
        # itself owns further deletion/upward walking. Keep node.parent intact.
        key = node.key.child_key(self.page_size)
        node.parent.children.pop(key, None)

    # real method reused verbatim (called unbound in helpers)
    collect_full_device_indices = UnifiedTreeCore.collect_full_device_indices


class TestMambaAnchorTombstoneCascade(unittest.TestCase):
    def setUp(self):
        self._saved = ct_mod._mamba_host_anchor_cache

    def tearDown(self):
        ct_mod._mamba_host_anchor_cache = self._saved

    def _flag(self, on: bool):
        ct_mod._mamba_host_anchor_cache = on

    def _chain(self):
        # root(dev) -> A(dev) -> B(mamba-host-only bridge) -> C(mamba-host-only)
        core = _FakeCore()
        core._register(core.root_node)
        a = _node(core.root_node, 10, full="device", mamba="device")
        b = _node(a, 20, full=None, mamba="host")
        c = _node(b, 30, full=None, mamba="host")
        for n in (a, b, c):
            core._register(n)
        return core, a, b, c

    def _delete_c_and_cascade(self, core, c):
        core._remove_leaf_from_parent(c)
        UnifiedTreeCore._iteratively_delete_tombstone_leaf(
            core, c, {ComponentType.FULL: 0, ComponentType.MAMBA: 0}, {}, {}
        )

    def test_anchor_flag_on_keeps_bridge_alive(self):
        self._flag(True)
        core, a, b, c = self._chain()
        self._delete_c_and_cascade(core, c)
        self.assertIn(b.key.child_key(1), a.children, "anchored bridge deleted")
        self.assertIsNotNone(b.component_data[ComponentType.MAMBA].host_value)

    def test_anchor_flag_off_deletes_bridge_upstream_style(self):
        self._flag(False)
        core, a, b, c = self._chain()
        self._delete_c_and_cascade(core, c)
        self.assertNotIn(
            b.key.child_key(1), a.children, "flag-off must match upstream cascade"
        )

    def test_collect_degrades_instead_of_crash_when_chain_broken(self):
        self._flag(True)
        core, a, b, c = self._chain()
        # b has no FULL device value: walking C->A crosses it (unrestorable).
        out = core.collect_full_device_indices(c.id, a.id)
        self.assertEqual(out.numel(), 0, "must degrade to empty restore")

    def test_collect_normal_chain_unaffected_flag_on(self):
        self._flag(True)
        core = _FakeCore()
        core._register(core.root_node)
        a = _node(core.root_node, 10, full="device")
        b = _node(a, 20, full="device")
        core._register(a), core._register(b)
        # until_node==a: only b's value is collected (root order).
        out = core.collect_full_device_indices(b.id, a.id)
        self.assertEqual(out.tolist(), [20])

    def test_collect_root_terminus_still_works(self):
        for on in (False, True):
            self._flag(on)
            core = _FakeCore()
            core._register(core.root_node)
            a = _node(core.root_node, 10, full="device")
            b = _node(a, 20, full="device")
            core._register(a), core._register(b)
            out = core.collect_full_device_indices(b.id, core.root_node.id)
            self.assertEqual(out.tolist(), [10, 20], f"flag={on}")


if __name__ == "__main__":
    unittest.main()
