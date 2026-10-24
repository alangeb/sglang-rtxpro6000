"""Defect #4 (prod :8001, 2026-10-24 17:32:31Z): a node id popped from the
device-eviction walk can name a node an earlier step of the SAME walk already
deleted. The scheduler died at UnifiedTreeCore.node_by_id with
``KeyError: 206628`` -- the id lived only in the walk's local cursor after the
tree had moved on (cascade delete / tombstone / split re-parent).

The tree itself is consistent after such a delete, so the sanity check cannot
see it: the stale id exists only inside the eviction walk. The fix skips the
stale step (the node's space is already back in the allocator) and counts it.

CPU-only, CI-registered.
"""

import unittest

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StaleCore:
    """tree_core double whose evict_device_leaf raises the production KeyError."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def evict_device_leaf(self, node_id, is_write_back):
        self.calls += 1
        raise self.exc


class _StaleCache:
    """Minimal stand-in exposing exactly what _evict_device_leaf touches."""

    def __init__(self, exc):
        self.tree_core = _StaleCore(exc)
        self.is_write_back = True
        self.metrics_collector = None
        self.stale_evict_skips = 0

    _free_values = staticmethod(lambda d, h: None)
    _accumulate_tracker = staticmethod(lambda t, r: None)

    _evict_device_leaf = UnifiedRadixCache._evict_device_leaf


class TestEvictStaleNodeGuard(unittest.TestCase):
    def test_key_error_is_skipped_and_counted(self):
        cache = _StaleCache(KeyError(206628))
        tracker = {}
        self.assertIsNone(cache._evict_device_leaf(206628, tracker))
        self.assertEqual(cache.stale_evict_skips, 1)
        self.assertEqual(cache.tree_core.calls, 1)

    def test_other_key_errors_also_skip_never_raise(self):
        # A KeyError from deeper inside the leaf step (arena lookup on a
        # cascade-deleted child) must degrade the same way: skip, count, keep
        # the scheduler alive.
        cache = _StaleCache(KeyError("node_by_id"))
        self.assertIsNone(cache._evict_device_leaf(7, {}))
        self.assertEqual(cache.stale_evict_skips, 1)

    def test_success_path_untouched(self):
        sentinel = object()

        class _Ok:
            def evict_device_leaf(self, node_id, is_write_back):
                class _R:
                    device_frees = {}
                    host_frees = {}
                    tracker = {}
                    mamba_host_invalidated = {}
                    backup_kv = sentinel

                return _R()

        cache = _StaleCache(None)
        cache.tree_core = _Ok()
        self.assertIs(cache._evict_device_leaf(1, {}), sentinel)
        self.assertEqual(cache.stale_evict_skips, 0)


if __name__ == "__main__":
    unittest.main()
