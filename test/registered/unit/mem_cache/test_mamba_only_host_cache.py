"""CPU coverage for the Mamba-only host tier gate (SGLANG_MAMBA_ONLY_HOST_CACHE).

The assembler builds host pools only when a hicache budget exists, because the
FULL KV host pool is the tree anchor every backup/load-back rides on. This
flag lets SGLANG_MAMBA_HOST_SIZE_GB build the host stack with a minimal KV
placeholder instead. Selection logic only -- no pools are allocated here.
"""

import unittest
from unittest import mock

from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
from sglang.srt.mem_cache.unified_cache import component_type as ct_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMambaOnlyHostCache(unittest.TestCase):
    def tearDown(self):
        ct_mod._mamba_only_host_cache_cache = None

    def _env(self, only: str, hicache: int):
        return mock.patch.multiple(
            ct_mod, _mamba_only_host_cache_cache=None
        ), mock.patch.dict(
            ct_mod.os.environ,
            {ct_mod.MAMBA_ONLY_HOST_CACHE_ENV: only},
            clear=False,
        ), mock.patch.object(
            hpa, "get_memory", return_value=mock.Mock(hicache_size=hicache)
        )

    def _gate(self, only: str, hicache: int, gb: float) -> bool:
        ct_mod._mamba_only_host_cache_cache = None
        with mock.patch.dict(
            ct_mod.os.environ, {ct_mod.MAMBA_ONLY_HOST_CACHE_ENV: only}
        ), mock.patch.object(
            hpa, "get_memory", return_value=mock.Mock(hicache_size=hicache)
        ):
            return hpa._mamba_only_host_cache(gb)

    def test_requires_explicit_gb_no_budget_and_flag(self):
        self.assertTrue(self._gate("1", 0, 8.0))
        self.assertFalse(self._gate("0", 0, 8.0))   # flag off
        self.assertFalse(self._gate("1", 6, 8.0))   # hicache budget exists
        self.assertFalse(self._gate("1", 0, 0.0))   # no explicit size

    def test_explicit_size_with_zero_budget_raises_without_flag(self):
        ct_mod._mamba_only_host_cache_cache = None
        with mock.patch.dict(
            ct_mod.os.environ,
            {
                ct_mod.MAMBA_ONLY_HOST_CACHE_ENV: "0",
                "SGLANG_MAMBA_HOST_SIZE_GB": "8",
            },
        ), mock.patch.object(
            hpa, "get_memory", return_value=mock.Mock(hicache_size=0)
        ):
            with self.assertRaises(ValueError) as cm:
                hpa._explicit_mamba_host_size_gb()
            self.assertIn("SGLANG_MAMBA_ONLY_HOST_CACHE", str(cm.exception))

    def test_explicit_size_with_zero_budget_ok_with_flag(self):
        ct_mod._mamba_only_host_cache_cache = None
        with mock.patch.dict(
            ct_mod.os.environ,
            {
                ct_mod.MAMBA_ONLY_HOST_CACHE_ENV: "1",
                "SGLANG_MAMBA_HOST_SIZE_GB": "8",
            },
        ), mock.patch.object(
            hpa, "get_memory", return_value=mock.Mock(hicache_size=0)
        ):
            self.assertEqual(hpa._explicit_mamba_host_size_gb(), 8.0)


if __name__ == "__main__":
    unittest.main()
