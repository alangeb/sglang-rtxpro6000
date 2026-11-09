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
- Rebuilt restart tree: /home/alangeb/sglang-rtxpro6000-cache/deploy/ (dir: /home/alangeb/sglang-rtxpro6000-cache/) (docker-compose.p316-prod.yaml + healthcheck.py + RUNBOOK.md + docker-compose.canary.yaml), generated from the LIVE container (inspect + /proc/1/cmdline archived in /home/alangeb/recovered-experiment-20261109/).
- RESTART = docker rm -f q38fn-p3-v3-prod && cd /home/alangeb/bench_out-rebuilt-20261109 && docker compose -f docker-compose.p316-prod.yaml up -d
- VERIFY after restart: /health -> restore_probe.py PASS (fresh boot) -> canary cycle OK (script: recovered-experiment-20261109/restore_canary.py).
- Model blobs verified complete post-restore: 206 safetensors shards, 125.9 GiB, 0 missing links.
- All v12 qualification/rollback/restore-probe commits live on branch q38fn/pennyroyal316-host-mamba (e250af9d89..d200a080b5) — merge that branch to fold docs+probes back.
