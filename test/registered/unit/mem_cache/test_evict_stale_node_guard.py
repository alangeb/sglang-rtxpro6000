"""Defect #4 (prod :8001, 2026-10-24 17:32:31Z): a node id popped from the
device-eviction walk can name a node an earlier step of the SAME walk already
deleted. The scheduler died at UnifiedTreeCore.node_by_id with
``KeyError: 206628`` -- the id lived only in the walk's local cursor after the
tree had moved on (cascade delete / tombstone / split re-parent).

The tree itself is consistent after such a delete, so the sanity check cannot
see it: the stale id exists only inside the eviction walk. The guard skips the
stale step (the node's space is already back in the allocator), absorbs ONLY
the arena miss (KeyError whose key equals the popped id -- foreign KeyErrors
re-raise), and counts consecutive skips so the caller's walk can abort a
non-converging pass (_STALE_EVICT_WALK_CAP).

CPU-only, CI-registered.
"""

import unittest

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _STALE_EVICT_WALK_CAP,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StaleCore:
    """tree_core double whose evict_device_leaf raises like the real arena."""

    def __init__(self, exc_factory):
        self.exc_factory = exc_factory
        self.calls = 0

    def evict_device_leaf(self, node_id, is_write_back):
        self.calls += 1
        raise self.exc_factory(node_id)


class _GuardCache:
    """Minimal stand-in exposing exactly what the guard path touches."""

    def __init__(self, exc_factory):
        self.tree_core = _StaleCore(exc_factory)
        self.is_write_back = True
        self.metrics_collector = None
        self.stale_evict_skips = 0
        self.stale_evict_skips_consecutive = 0

    _free_values = staticmethod(lambda d, h: None)
    _accumulate_tracker = staticmethod(lambda t, r: None)

    _evict_device_leaf = UnifiedRadixCache._evict_device_leaf


class TestEvictStaleNodeGuard(unittest.TestCase):
    def test_arena_miss_is_skipped_and_counted(self):
        # Exactly the production shape: KeyError(node_id) from the arena.
        cache = _GuardCache(lambda nid: KeyError(nid))
        self.assertIsNone(cache._evict_device_leaf(206628, {}))
        self.assertEqual(cache.stale_evict_skips, 1)
        self.assertEqual(cache.stale_evict_skips_consecutive, 1)

    def test_foreign_key_error_re_raises(self):
        # A KeyError from deeper logic (key != popped id) is a different bug
        # and must not be laundered into a skip.
        cache = _GuardCache(lambda nid: KeyError("something_else"))
        with self.assertRaises(KeyError):
            cache._evict_device_leaf(7, {})
        self.assertEqual(cache.stale_evict_skips, 0)

    def test_key_error_without_args_re_raises(self):
        cache = _GuardCache(lambda nid: KeyError())
        with self.assertRaises(KeyError):
            cache._evict_device_leaf(7, {})
        self.assertEqual(cache.stale_evict_skips, 0)

    def test_consecutive_counter_resets_on_clean_step(self):
        sentinel = object()

        class _Flaky:
            def __init__(self):
                self.calls = 0

            def evict_device_leaf(self, node_id, is_write_back):
                self.calls += 1
                if self.calls == 1:
                    raise KeyError(node_id)

                class _R:
                    device_frees = {}
                    host_frees = {}
                    tracker = {}
                    mamba_host_invalidated = {}
                    backup_kv = sentinel

                return _R()

        cache = _GuardCache(None)
        cache.tree_core = _Flaky()
        self.assertIsNone(cache._evict_device_leaf(1, {}))
        self.assertEqual(cache.stale_evict_skips_consecutive, 1)
        self.assertIs(cache._evict_device_leaf(2, {}), sentinel)
        self.assertEqual(cache.stale_evict_skips_consecutive, 0)
        self.assertEqual(cache.stale_evict_skips, 1)

    def test_walk_cap_constant_is_sane(self):
        # The cap lives in the caller's loop; the guard's contract is only to
        # count. Pin the constant so an accidental 0/1 regression is caught.
        self.assertGreaterEqual(_STALE_EVICT_WALK_CAP, 8)


class _LoopCore:
    """tree_core double: the walk pops ids forever, every leaf step is stale."""

    def __init__(self):
        self.start_calls = 0
        self.end_calls = 0
        self.popped = 0

    def evict_device_start(self, ct, request_cnt):
        self.start_calls += 1

    def evict_device_end(self, ct):
        self.end_calls += 1

    def evict_device_leaf(self, node_id, is_write_back):
        raise KeyError(node_id)  # the production arena miss


class TestEvictWalkAbortsOnStaleStreak(unittest.TestCase):
    """The caller's loop, not the guard, owns the spin bound.

    A walk that pops nothing but dead ids must terminate (bounded pops) and
    must still run evict_device_end, so the shortfall surfaces as a normal
    allocation retry instead of a scheduler spinning inside one eviction call.
    """

    def _cache(self):
        core = _LoopCore()

        class _C:
            tree_components = (ComponentType.FULL,)

            def __init__(self):
                self.tree_core = core
                self.is_write_back = True
                self.metrics_collector = None
                self.stale_evict_skips = 0
                self.stale_evict_skips_consecutive = 0

            def _evict_device_next_node(self, ct, tracker):
                core.popped += 1
                return core.popped, True  # always a candidate, always progress

            _evict_device_leaf = UnifiedRadixCache._evict_device_leaf
            _evict_components = UnifiedRadixCache._evict_components

        return _C(), core

    def test_walk_terminates_and_closes_cleanly(self):
        cache, core = self._cache()
        tracker = {ct: 0 for ct in cache.tree_components}
        cache._evict_components(
            {ct: 10 for ct in cache.tree_components}, tracker, None
        )
        # Terminated at the cap, not at the request: bounded, and the pop
        # that tripped the cap is one past it.
        self.assertLessEqual(core.popped, _STALE_EVICT_WALK_CAP + 2)
        self.assertGreater(core.popped, _STALE_EVICT_WALK_CAP // 2)
        # The finally still ran: the walk is left in a clean state.
        self.assertEqual(core.end_calls, core.start_calls)
        self.assertGreater(cache.stale_evict_skips, _STALE_EVICT_WALK_CAP)


if __name__ == "__main__":
    unittest.main()
