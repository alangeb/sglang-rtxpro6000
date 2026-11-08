# Production pin — 2026-11-08

| Item | Value |
|---|---|
| Git tag (annotated, on origin) | `q38fn-p3-v11-prod-20261108` @ `13f3f83a6` |
| Container | `q38fn-p3-v3-prod` on cuda, port 8001→30000, healthy since 2026-11-04 |
| Image | `sglang-rtxpro6000:q38fn-p3-mamba-only-v11` (ID `81fcf050e2d2`, built 2026-10-24T21:33+02) |
| Compose files | `/home/alangeb/bench_out/q38fn-p3-experiment-20261011/docker-compose.prod-p3.yaml` + `docker-compose.v11-flip.yaml` (flip layer swaps v9→v11 image only) |
| Launch | `bash /experiment/launch_variants_v2.sh` (VARIANT=A, MODEL_SOURCE=radix), env: MAMBA_ONLY_HOST_CACHE/HOST_ANCHOR/COMPLETION_RETENTION/PER_SLOT=1, SPILL_LOW_WATER=8, HOST_SIZE_GB=10, TRACK_INTERVAL=4096 |
| Container snapshot | `q38fn-prod-live-snapshot:20261011` exists (older, v256-era) |

## Restore procedure
1. Code: `git checkout q38fn-p3-v11-prod-20261108` (tree verified byte-identical to running container's `python/sglang` except Triton autotune JSONs).
2. Deploy: `cd /home/alangeb/bench_out/q38fn-p3-experiment-20261011 && docker compose -f docker-compose.prod-p3.yaml -f docker-compose.v11-flip.yaml up -d --force-recreate` — the v11 image still exists locally; rollback chain: v11 → v9 (`q38fn-p3-mamba-only-v9` == `prod-live-20261022`, pinned) → v256.
3. Never delete images `q38fn-p3-mamba-only-v11`, `-v9`, `prod-live-20261022`.

## Rebase plan (in progress)
- Target: `pennyroyal-v3.1.6` (upstream `jpezzulli/sglang-rtxpro6000`, added as remote `upstream`; 263 commits over our base v3.0.0 `5d33142c1`; our merge-base is `b9f01d067`).
- Our patch set = 20 commits `pennyroyal-v3.1.6..HEAD` (all mamba host-cache work + docker health fix + README).
- Expected collisions (only these two): `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`, `python/sglang/srt/mem_cache/unified_radix_cache.py`.
- Worthwhile upstream picks folded in: v3.1.5 chunked mamba temporal copy (`memory_pool.py`, clean), v3.1.6 hybrid-prefetch fallback fix (`unified_radix_cache.py`), v3.0.1 PR#73 host slot stride page-alignment (`pool_host/mamba.py`, clean — latent until L3/NIXL used).
- Rebase branch: `q38fn/pennyroyal316-host-mamba` (prod branch stays untouched until rebase is fully validated on the scratch compose).

## Rebase status — RC1 (2026-11-08, same day)
- Branch `q38fn/pennyroyal316-host-mamba` @ tag `q38fn-p316-rebase-rc1` (peels to `1345f14b62`, the exact code the candidate image was built from; branch tip adds docs only): 20 fork commits replayed on `pennyroyal-v3.1.6`; worktree `/home/alangeb/wt-q38fn-rebase-316`; tree clean, zero real conflict markers, 11/14 fork files byte-identical to prod branch, 3 merged (Dockerfile install-strategy change, assembler + unified_radix_cache = both-side changes, our hunks + upstream v3.1.x content verified present).
- Tests (mem_cache suite, inside v3.1.6 image, CPU-only box): rebased FAILED-set vs pristine-v3.1.6 FAILED-set delta = +4/-0. The 4 (`test_unified_radix_allocation_eviction.py`) fail IDENTICALLY on pristine upstream v3.1.6 and on prod-branch code — environment-related (no GPU), not caused by our patches. All 10 fork mamba unit files: 104 passed.
- Candidate image built: `sglang-rtxpro6000:q38fn-p316-mamba-only-v1` (7b1350112c51, label revision 1345f14b62) = overlay of rebased python/ onto official `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.6` (31936dafd2ba), import gate passed.
- REMAINING before cutover: (1) GPU boot + serving smoke on scratch compose (needs GPU-0 quiet window — one-model-per-GPU rule; prod :8001 untouched); (2) bench parity vs v11 (llama-benchy + forkN); (3) miss-probe regression (wiki q38fn-cache-miss-investigation); (4) flip prod overlay to new tag; (5) re-pin this file + wiki to the v3.1.6 image.
## GPU smoke test — PASSED (2026-11-08, operator-approved GPU-0 window)
- Prod stopped 08:45Z (graceful, oom=false), smoke container q38fn-p316-rc1-smoke (image q38fn-p316-mamba-only-v1) booted on GPU 0, port 30001. Healthy after 390 s.
- Recipe identical to prod (variant A/radix, 0.981, 32 device slots, 10.03 GB host tier page_first, track 4096, D6 guard ACTIVE, mamba_total=170). Zero tracebacks in log. RAM avail never below 26 GB; GPU 91.7/97.9 GB.
- Functional: chat completion content+reasoning OK (finish=stop), prefix cache serving (cached_tokens=64 on repeat), mamba match + [MAMBA-HOST] D6 logs live, /metrics mamba counters live.
- Mini-bench (llama-benchy, bs=1): pp2048 8159 t/s (TTFR 229 ms), tg32 203 t/s — same ballpark as v11 prod (ITL p50 4.1 ms / TTFT p50 444 ms measured on 62k-token agentic traffic; not directly comparable, parity bench still pending).
- Compose kept for re-run: /home/alangeb/bench_out/q38fn-p316-smoke-20261108/docker-compose.smoke.yaml (entrypoint:[] + GPU reservation are REQUIRED — base v3.1.6 image ships ENTRYPOINT pennyroyal; omitting the reservation fails with "No accelerator").
- Prod restored: docker start q38fn-p3-v3-prod → healthy after 330 s, mamba tier reallocated, restarts=0.
- REMAINING to cutover: bench parity vs v11, miss-probe regression, prod overlay flip, re-pin.
