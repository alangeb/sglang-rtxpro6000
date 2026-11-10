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

## INCIDENT UPDATE 2026-11-09 (authoritative for restarts)
The bench_out compose/launcher tree was lost from /home during the VM rebuild + restore (bench_out/ now holds only root-owned empty mount skeletons). Production runs correctly but had NO restart path until reconstruction.
- Rebuilt restart tree: /home/alangeb/sglang-rtxpro6000-cache/deploy/ (dir: /home/alangeb/sglang-rtxpro6000-cache/) (docker-compose.p316-prod.yaml + healthcheck.py + RUNBOOK.md + docker-compose.canary.yaml), generated from the LIVE container (inspect + /proc/1/cmdline archived in /home/alangeb/sglang-rtxpro6000-experiments/).
- RESTART = docker rm -f q38fn-p3-v3-prod && cd /home/alangeb/sglang-rtxpro6000-cache/deploy && docker compose -f docker-compose.p316-prod.yaml up -d
- VERIFY after restart: /health -> restore_probe.py PASS (fresh boot) -> canary cycle OK (script: sglang-rtxpro6000-experiments/restore_canary.py).
- Model blobs verified complete post-restore: 206 safetensors shards, 125.9 GiB, 0 missing links.
- All v12 qualification/rollback/restore-probe commits live on branch q38fn/pennyroyal316-host-mamba (e250af9d89..d200a080b5) — merge that branch to fold docs+probes back.

## NVMe PLE MODE ENABLED 2026-11-09 ~14:00Z (config change, image unchanged: q38fn-p316-mamba-only-v1)
Prod now streams the 47.68 GiB PLE table from /data/flash-next-ple (ssd_stream reader, io_uring/O_DIRECT, page cache = row cache) instead of pinning it in host RAM. Host RAM freed: 65 -> 17 GiB used (available 32 -> 80 GiB); scheduler RSS 60.5 -> 13.3 GiB. GPU unchanged (91.6 GiB).
- compose: /home/alangeb/sglang-rtxpro6000-cache/deploy/docker-compose.p316-prod.yaml now has: PENNY_PLE_BACKEND=nvme, PENNY_PLE_NVME_MODEL=/data/flash-next-ple, SGLANG_PLUGINS=ssd_stream, PYTHONPATH=/opt/pennyroyal/.ple-nvme:/opt/pennyroyal/python, /data:/data:ro mount, seccomp=unconfined, model-path=/data/flash-next-ple (the PREPARED OVERLAY, not the HF snapshot — its weight shards symlink into /data/cache/huggingface blobs; PLE tensors excluded from its index).
- Pre-req checks passed: ssd-stream.json source config/index sha256 == current HF snapshot blobs (exact match); check_ple_nvme.py preflight OK -> manifest sha 9874441415594e226984f9bc47f5a80513109a43ddd17b16fd794d7290a1a879. Boot log: "SSD Stream enabled: backing=/data/flash-next-ple/ple/layer-0.bin".
- Perf (direct /generate timing): warm-rows prefill 200k tok = 10,459 tok/s (RAM baseline 11,100, within noise); decode 211 tok/s (baseline ~201, parity); BRAND-NEW 200k doc first pass = 4,706 tok/s (~23 s of page-in I/O, one-time per document, ~0.11 s per 1k tokens).
- ROLLBACK to RAM PLE: in the compose set PENNY_PLE_BACKEND=ram (or remove PENNY_PLE_* + SGLANG_PLUGINS + PYTHONPATH overrides), revert --model-path to /root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594, drop the /data:/data:ro mount; restart.
- NOTE: restore_probe.py was lost with bench_out and is not yet recreated; the restore-canary (sglang-rtxpro6000-cache/logs/canary.log) covers the same assertions every 15 min and passes in NVMe mode (recall true, invalidated 0).

## IDLE-CPU SPIN FIXED 2026-11-09 ~16:05Z (--sleep-on-idle)
Symptom: q38fn-p3-v3-prod pinned 94-100% of one core while completely idle (GPU 0%, zero requests) — measured since the v3.1.6-era boots, quantified 2026-11-09.
Root cause (py-spy on sglang::scheduler, container pid 67): Scheduler.event_loop_overlap -> Scheduler.on_idle (scheduler.py:4333-4345) runs a FULL UnifiedTreeCore.sanity_check (recursive walk, unified_tree_core.py:2431/2546, _is_host_leaf per node), pool invariant check, and ShmLoadSnapshotWriter.write + SchedulerLoadInquirer.get_loads on EVERY idle iteration, and `maybe_sleep_on_idle` was a no-op because IdleSleeper is only created when server_args.sleep_on_idle is true (scheduler.py:811 -> server_args.py:1323, default False). NOT caused by the NVMe PLE flip — same behavior existed since the v3.1.6 cutover; the ssd-stream worker threads were idle in every dump.
Fix: launch flag `--sleep-on-idle` added to docker-compose.p316-prod.yaml (parks the event loop in zmq poller up to 1 s when idle; wake on request is immediate). Verified: container CPU 100.74% -> 0.47-0.53% idle; generation OK (17*23=391); memory unchanged 14.2 GiB. Rollback: remove the flag.

## PLE ROW-ID TRACE LIVE 2026-11-09 ~20:07Z (patched plugin via bind-mount, image unchanged)
sglang-rtxpro6000-cache/deploy/ple-plugin/ (copy of the in-image 0.2.0+pennyroyal2 adapter + _BatchTraceWriter)
is bind-mounted ro over /opt/pennyroyal/.ple-nvme; SGLANG_SSD_STREAM_TRACE_PATH=/trace/ple-rowids.bin writes
per-batch UNIQUE row-id records ([int32 n][n x int64]) to sglang-rtxpro6000-experiments/ple-trace/.
Purpose: 1-2 h of real agent traffic -> ple_trace_sim.py (same dir) -> hit-rate curves for LRU / LRU+admission /
static-hot at 0.25-16 GiB -> pick policy+size for the pinned row cache (the ~10% NVMe-PLE cost + page-cache
eviction risk under a co-resident second model). Boot verified: "SSD Stream row-id trace enabled" + reader
engaged; generation OK; health 200; idle CPU 1.5% (sleep-on-idle intact). Rollback: remove the ple-plugin mount
+ the TRACE_PATH env, restart (image untouched throughout).

## PLE ROW CACHE LIVE 2026-11-10 ~11:00Z (final production config)
Prod runs: SGLANG_SSD_STREAM_ROW_CACHE_GIB=4 (pinned exact-LRU row cache, 26.8M rows), SGLANG_SSD_STREAM_NO_PAGE_CACHE=1
(fadvise DONTNEED after every gather -> PLE access NEVER benefits from page cache), trace still on. Patched plugin
bind-mounted from sglang-rtxpro6000-cache/deploy/ple-plugin; source committed to tools/ple_nvme (rebase branch).
GEOMETRY CORRECTION: the table is 320,001,536 rows x 160 B (16 heads x 20M x 160B) — NOT 20M x 2560B. The observed
working set is only ~2.55M rows = 388 MiB; 4 GiB pin covers it 10x over (1 GiB would suffice; env knob to resize).
Benchmarks (200k-token direct tests, page cache bypassed, spin fixed):
- floor (no cache, bypass): cold 67.4 s / warm 27.6 s (drive-state dependent)
- 1 GiB: docA_cold 20.1 s / docB_shuffled 20.3 / docA_warm 20.3 (all ~9.9k tok/s)
- 2 GiB: docA_cold 41.8* / docB 20.0 / docA_warm 20.0   (* first-big-read transient + host I/O contention)
- 4 GiB: docA_cold 54.2* / docB 20.3 / docA_warm 20.2; recheck: cold 40.6 -> warm 19.9 (2x on repeats, at compute floor)
- Correctness: 3/3 prompts any-of-3 match baseline at every size; outputs always fluent (nondeterminism = batch-state
  sampling flips, established separately). Decode 219-245 tok/s all configs. 5090/vLLM untouched throughout.
- CONCLUSION: on the pro6000 the streaming I/O is compute-hidden (cold ~10k tok/s ~= RAM baseline 11.1k); the cache
  buys determinism (page-cache independence + 2x on repeat prefills vs cold) but no steady-state gain. The earlier
  "10% impact" was the idle-CPU spin. Keep 4 GiB (headroom); drop to 1 via env if the second model needs RAM.
- First-big-read-after-boot is variable (20-54 s for 200k) and NOT cache-related (reproduces at all sizes).
