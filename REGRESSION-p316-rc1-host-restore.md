# REGRESSION: p316 RC1 candidate loses host-tier mamba restore under slot pressure

**Date:** 2026-11-08 · **Verdict: NOT QUALIFIED — v11 stays production.**
Candidate `sglang-rtxpro6000:q38fn-p316-mamba-only-v1` (rebase of our 20 commits onto
`pennyroyal-v3.1.6`) fails the GPU-level restore probe that production v11 passes.
Everything else qualified: boot, serving, cache serving, metrics, bench parity, 104 unit tests.

## Probe (bench_out/q38fn-p316-smoke-20261108/restore_probe.py)
R1: 5448-token prompt with secret, greedy → warm answer.
44 distinct ~4.4k-token prompts → device mamba pool (32 slots) exhausted → low-water spill fires.
R2: identical prompt → expect host-tier mamba restore (`mamba_host=1`), answer unchanged.
R3: `/flush_cache` + identical prompt → cold oracle.

## Evidence (same probe, same hour, same GPU)

| | v11 (prod) | p316 RC1 candidate |
|---|---|---|
| R2 log line | `input=5448 full_kv=4096 device=4096 host_hit=0 mamba_host=1` | `input=5448 full_kv=3520 device=3520 host_hit=0 mamba_host=0` |
| R2 counters | `load_back_tokens_total{pool=mamba}` 0→**2** (post-restart delta) | restored counter absent (0) |
| R2 behaviour | node kept alive after spill (host-anchor), checkpoint pulled back from host tier, KV reusable to the 4096 grid | spilled checkpoint's node **not kept alive**: match truncated to 3520, mamba recomputed 0→3520, host copy orphaned |
| Semantic result | secret recalled | secret recalled (full recompute is correct) — but the fork's core feature silently did not engage |
| spill events | spilled, then restored | `spill: issued 8/8` fired, 16 slots demoted, **0 restored** |

## Why unit tests missed it
All 104 fork unit tests pass on the rebased tree — they exercise the component logic in
isolation. The breakage is an *interaction*: under v3.1.6's rewritten allocation/eviction
stack (`kv_cache_configurator.py`, `allocator/swa.py`, auto KV sizing, hybrid-prefetch
revoke), the post-spill node loses its device anchor and becomes tree-evictable, so the
host copy is unreachable. Suspects to bisect: `evict_for_alloc` ordering, lock-ref
accounting on spill, `_mamba_only` sizing path vs v3.1.6 `build_draft_swa_entry` changes
in `hybrid_pool_assembler.py` (one of the two genuinely-merged files).

## Disposition
- Production remains `q38fn-p3-v3-prod` on `q38fn-p3-mamba-only-v11` (tag
  `q38fn-p3-v11-prod-20261108`), healthy, restarts=0, restart policy intact. No flip performed.
- Candidate image kept for debugging: `q38fn-p316-mamba-only-v1` (`7b1350112c51`).
- Re-run harness: `restore_probe.py` (now a regression test) + `docker-compose.smoke.yaml`.
- Next: bisect v3.1.6 allocator/eviction changes against `mamba_only_host_cache` +
  `SGLANG_MAMBA_HOST_ANCHOR` paths; re-run probe per bisect step; only then re-qualify.
