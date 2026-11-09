# NVMe-Backed PLE in Production (q38fn, pro6000)

**Status:** LIVE since 2026-11-09 ~14:00Z · Image unchanged: `sglang-rtxpro6000:q38fn-p316-mamba-only-v1` (prod pin `q38fn-p316-prod-20261108`)
**Authoritative config:** `/home/alangeb/sglang-rtxpro6000-cache/deploy/docker-compose.p316-prod.yaml`

## 1. What this is

The Qwen3.8-Flash-Next PLE table (20 M rows × 2560 B fp8, layer 2, n-gram-row addressed) is
**47.68 GiB**. Until 2026-11-09 it was pinned in host RAM (`PENNY_PLE_BACKEND=ram` default), the
single largest RAM consumer on the box. It is now **streamed from NVMe** by the `ssd_stream`
reader (io_uring + O_DIRECT, bounded row buffers + page pool; the Linux page cache acts as the
row cache). Prepared snapshot: `/data/flash-next-ple/` (`ple/layer-0.bin`, 51,200,245,760 bytes,
+ `ssd-stream.json` manifest, + all ordinary weights as **absolute symlinks into
`/data/cache/huggingface/...` blobs** — hence `/data` must be mounted ro at the same host path).
Upstream feature docs: `NVME-PLE.md`, `tools/ple_nvme/` (in-repo).

## 2. Exact change (compose deltas)

| item | value |
|---|---|
| `--model-path` | `/data/flash-next-ple` — **the prepared overlay, NOT the HF snapshot** (its index excludes PLE tensors) |
| env | `PENNY_PLE_BACKEND=nvme`, `PENNY_PLE_NVME_MODEL=/data/flash-next-ple`, `SGLANG_PLUGINS=ssd_stream`, `PYTHONPATH=/opt/pennyroyal/.ple-nvme:/opt/pennyroyal/python` |
| mounts | `/data:/data:ro` added; everything else unchanged |
| security | `seccomp=unconfined` (io_uring; same as upstream `run.sh --nvme-ple`) |
| user | `user: "0"` (v3.1.6 base has `/root` 750; default user `penny` cannot traverse `/root/.cache`) |
| network | **bridge + `ports: ["8001:30000"]`** — NOT host networking |
| other | `ipc: host`, `shm_size: 32g`, `security_opt label=disable` (unchanged from prod) |

No `--field ple_*` args: those belong to the NIXL namespace helper only (`configs/pennyroyal/ple-backend.sh`),
and `sglang serve` rejects them (we proved it the hard way).

## 3. Pre-flight that gated the flip

- `ssd-stream.json` source identity vs current HF snapshot blobs: `config_sha256`/`index_sha256`
  **byte-identical match**.
- `scripts/pennyroyal/check_ple_nvme.py --source <HF snapshot> --prepared <overlay>` →
  manifest sha `9874441415594e226984f9bc47f5a80513109a43ddd17b16fd794d7290a1a879`. (Run it with
  `--source` spelled as the **manifest-recorded path**; the check compares literal resolved paths,
  not just hashes.)
- `BF16-STAGING-HOTFIX.md` is v2.5.2-only — **not applicable** on v3.1.6.
- Boot log must show: `SSD Stream enabled: backing=/data/flash-next-ple/ple/layer-0.bin …`.

## 4. Measured results (200k-token direct `/generate` tests, device idle)

**Memory (the prize):**

| metric | RAM mode | NVMe mode |
|---|---|---|
| host RAM used / available | 65 / 32 GiB | **17 / 80 GiB** |
| scheduler RSS | 60.5 GiB (incl. 47.68 GiB pinned table) | **13.3 GiB** (biggest map 8.96 GiB) |
| GPU | 91.7 GiB | 91.6 GiB (unchanged) |
| page cache | 42 GiB | 90 GiB (holds the table; reclaimable) |

**Performance:**

| scenario | NVMe | RAM baseline | verdict |
|---|---|---|---|
| prefill, rows warm in page cache | 10 459 tok/s | 11 100 tok/s | parity (within noise) |
| decode | 211 tok/s | ~201 tok/s | parity |
| **brand-new 200k doc, first pass** | 4 706 tok/s (~23 s page-in) | — | one-time cost, **~0.11 s per 1k tokens**, then cached |

**Underlying NVMe** (2026-11-09 idle-device bench vs `layer-0.bin`, script in
`sglang-rtxpro6000-experiments/nvme_read_bench_20261109.py`): O_DIRECT seq 2.68 GiB/s;
rand 4 K QD1 p50 87 µs / p99 126 µs; QD32 66.4 k IOPS (259 MiB/s) — sweet spot QD16–32;
rand 2.75 MB (expert page) 557 µs @ 5.03 GiB/s (QD8). The old virtiofs `/data` was 2.8 k IOPS
with O_DIRECT rejected — the passthrough NVMe is what made this feature viable.

## 5. Operations

**Restart** (unchanged procedure, now NVMe-aware):
```
docker rm -f q38fn-p3-v3-prod
cd /home/alangeb/sglang-rtxpro6000-cache/deploy && docker compose -f docker-compose.p316-prod.yaml up -d
```
Boot ~7–12 min (JIT caches warm in `../cache/`; first boot after cache loss recompiles).
**Verify:** `/health` 200 → generation sane → log contains `SSD Stream enabled` → scheduler RSS
≈13 GiB (not 60) → canary cycle OK (`../logs/canary.log`, container `q38fn-restore-canary`).

**Rollback to RAM PLE** (one restart):
1. `PENNY_PLE_BACKEND=ram` (or delete the four PENNY/SGLANG/PYTHONPATH env additions)
2. `--model-path` back to the HF snapshot
   `/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594`
3. drop the `/data:/data:ro` mount · restart.

**Monitoring:** `free -g` (used should stay ≈17–20 GiB; growth = leak), canary JSONL
(`ok`, `recall`, `invalidated` fields), `/metrics` mamba counters. Restore probe script was lost
with bench_out — canary covers the same assertions; recreate on request.

## 6. Gotchas (all learned live on 2026-11-09)

1. Plugin import needs `PYTHONPATH=/opt/pennyroyal/.ple-nvme` + `SGLANG_PLUGINS=ssd_stream`
   (isolated `--target` install is not on sys.path; without them the server **silently** stays in
   RAM mode — detect via scheduler RSS, it fails closed only if the plugin *loads and errors*).
2. `--field` args are NIXL-namespace-helper syntax, not `sglang serve` flags.
3. The prepared overlay resolves weights through **absolute** symlinks → mount `/data` at the
   identical host path, ro.
4. Old prod ran as root (`user: "0"`); the image's `/root` is 750 → default user breaks model
   config read.
5. Prod networking is **bridge + 8001→30000**, not host mode (server silently healthy on 30000
   while :8001 is dead).
6. `PENNYROYAL_PLE_BACKEND` (old name) is a no-op; only `PENNY_PLE_BACKEND` is read
   (`tools/ple_nvme/ssd_stream/src/sglang_ssd_stream/plugin.py`).

## 7. Context: how this config came to be (2026-11-09)

Host disk was replaced with a passthrough NVMe mid-day (old disk at `/data2`); a concurrent
`rsync /data2/ → /data/` restore wiped nothing but required re-verification (model blobs 206/206
shards present, 125.9 GiB). During the rebuild the original `bench_out` compose/launcher tree was
lost from `/home` — production was reconstructed from the live container (`docker inspect` +
`/proc/1/cmdline` archived in `sglang-rtxpro6000-experiments/`) into
`sglang-rtxpro6000-cache/deploy/`, which is why all paths above live there. Full incident record:
`PROD_PIN.md` (incident + NVMe sections) and the tau wiki topic
`q38fn-cache-miss-investigation`.
