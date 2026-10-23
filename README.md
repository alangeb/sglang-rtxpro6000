# Q38-FN — Penny Royal fork with Mamba-only host cache (production)

> **This repository is a fork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
> ("Penny Royal"). The default branch, `q38fn/pennyroyal3-host-mamba`, is the source of the
> container that serves production traffic on our RTX PRO 6000. The unmodified upstream
> Penny Royal README is preserved at the [end of this file](#upstream-penny-royal-readme-unmodified).**

## What this is

A production SGLang runtime for **Qwen3.8-Flash-Next (NVFP4)** on **one RTX PRO 6000 Blackwell
(96 GB, SM120)**, carrying 19 commits of local work on top of Penny Royal 3.0
(`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.0.0`, source revision `5d33142c1f`).

The work is concentrated in one subsystem: **the Mamba/GDN recurrent-state checkpoint tier** of
the unified radix cache (`python/sglang/srt/mem_cache/`). The model is hybrid — 48 of 64
layers are Gated DeltaNet linear attention, whose state is a fixed-size recurrent tensor rather
than growing KV. Penny Royal 3.0 could spill those states to host memory, but only under an
upstream invariant that made the spill path almost never fire in our configuration. This branch
removes that blocker and makes the host tier actually usable.

## Why: the problem this solves

Agentic traffic is the opposite of chat traffic. One conversation runs to ~90–130K prompt
tokens, and almost every turn re-uses the whole prefix. The cost of a turn is therefore almost
entirely decided by **how much of the prefix the engine has to recompute**, and for a hybrid
model that is bounded by the *Mamba checkpoint grid*, not by the KV pool.

Upstream keeps a tree node alive only if its **FULL** (attention-KV) copy survives on some layer.
Our configuration deliberately keeps almost no host KV — the KV host tier is a
`MINIMAL_KV_HOST_GB` placeholder — so a spilled Mamba checkpoint sits on a node whose FULL host
copy does not exist. Upstream's rule then made that node, and its checkpoint, immediately
evictable: the spill path fired, wrote, and the state was dropped before anything could read it.
Measured before this work: ~0 % host-tier contribution, and every long turn recomputed its whole
prompt.

## What changed

| Change | Flag / commit | Effect |
|---|---|---|
| **Mamba-only host cache** — relax the "FULL host copy required" invariant so a node may stay anchored on Mamba-host evidence alone | `SGLANG_MAMBA_ONLY_HOST_CACHE=1`, `316f4e3c33` | The host tier becomes reachable at all. Without it the rest of this table is inert |
| **Host anchor** — `host_anchored()` used wherever upstream *requires* FULL to keep a node alive (never to claim a node is restorable) | `SGLANG_MAMBA_HOST_ANCHOR=1`, `ee55aaac77` | Checkpoint survives eviction pressure |
| **Proactive spill at the allocation choke point** — device→host spill when the Mamba slot pool hits a low-water mark, instead of only on eviction | `SGLANG_MAMBA_SPILL_LOW_WATER=8`, `e008836b3f` + fix `0e9d10a0f9` | Spill happens before the pool is exhausted, off the critical path |
| **Completion-boundary retention** — prefer to keep the checkpoint taken at a turn boundary | `SGLANG_MAMBA_COMPLETION_RETENTION=1`, `cba883d77c` | The one checkpoint an agent will actually re-enter is the one kept |
| **Per-slot guard + accounting** — reject out-of-range slot ids before the copy; count checkpoints destroyed by KV eviction | `SGLANG_MAMBA_PER_SLOT=1`, `02bbc25df8`, `204c83c500` | Turns an illegal-memory-access crash into a host miss |
| **Spill-lock release** (HEAD) | `11d16eb1df` | Releases the spill lock before recycling the spilled device slot |

**Net effect on production:** effective prefix-cache hit ratio **96.96 %** across 1.48 B prompt
tokens in 48 h (device 96.39 %, host 0.57 %), ITL p50 **4.1 ms**, TTFT p50 **444 ms**, at an
average prompt of **62,563 tokens**. The host tier now demonstrably serves reads
(`host_hit=8512 mamba_host=1 branching=Yes` in engine logs).

## How to use it

The launch recipe is **not** in this repository (see *Runtime recipe* below). The shape of a
production launch:

```bash
SGLANG_MAMBA_ONLY_HOST_CACHE=1   # required for everything below
SGLANG_MAMBA_HOST_ANCHOR=1
SGLANG_MAMBA_COMPLETION_RETENTION=1
SGLANG_MAMBA_SPILL_LOW_WATER=8   # spill when free device slots drop to this
SGLANG_MAMBA_HOST_SIZE_GB=10     # host Mamba pool size

sglang serve \
  --model-path <RadixArk/Qwen3.8-Flash-Next-NVFP4 snapshot> \
  --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
  --kv-cache-dtype fp8_e4m3 --page-size 64 \
  --context-length 200000 --max-total-tokens 1000000 \
  --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
  --chunked-prefill-size 4096 --mem-fraction-static 0.981 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-track-interval 4096 --mamba-ssm-dtype bfloat16 \
  --max-mamba-cache-size 32 --gdn-mtp-cache-mode none \
  --speculative-num-steps 3 --speculative-num-draft-tokens 3
```

**Checkpoint density is the knob that matters.** `--mamba-track-interval` sets one checkpoint
every N tokens; the pool is `--max-mamba-cache-size` device slots plus the host tier (10 GB →
170 slots at 59 MB/slot). At interval 4096 that is **202 checkpoints ≈ 827K tokens of rewind
depth**, and it is what a long agent conversation actually consumes: 8 concurrent sessions at
~90K tokens need ~168 of the 170 host slots. Halving the interval doubles the rewind depth
resolution and halves the coverage; at interval 64 a checkpoint costs ~922 KB per KV-token
equivalent instead of ~4.9 KB, which is what made the tier useless before this work.

`--gdn-mtp-cache-mode none` is deliberate: with speculative decoding enabled, the draft path
would otherwise force a Mamba checkpoint per draft width. `--cuda-graph-max-bs-decode 8` keeps
the C=1 batch in the bucket the adaptive MTP widths were tuned for.

## Known issues on this branch

- **`sanity_check` false positive.** The idle invariant *"node backed up but parent not backed
  up"* (`unified_tree_core.py:2392`) has no Mamba carve-out, so in Mamba-only host mode a child
  can legitimately hold a FULL host copy its parent never got. It has taken the scheduler down
  three times (2026-10-14/15/16). Upstream addressed the neighbouring host-LRU check in
  `ad09a8171d`, then reverted it (`f95229f9f9`). Not yet fixed here.
- **Tombstone-cascade guard missing.** `00e50f64e0` (*keep host-anchored bridge nodes alive in
  tombstone cascade*) is on branch `mamba-host-anchor-v253` and **not** on this branch. Without
  it a cascade can delete an ancestor an in-flight match stands on → `AttributeError` on
  `collect_full_device_indices` → scheduler exit 137. Cherry-pick before the next deploy.
- **Healthcheck port.** The image's baked probe targets `:8001`; this deployment publishes
  `30000→8001`. Use the self-resolving `docker/pennyroyal/healthcheck.py` (`5dac6b0790`) and set
  `PENNYROYAL_HEALTH_PORT`, or Docker will report a healthy server as `unhealthy`.

## Runtime recipe is not in git

The compose file, launcher, `.env`, adaptive-width JSON and host cache directories live in
`~/bench_out/q38fn-p3-experiment-20261011/` on the serving host, outside version control. The
image tag we run (`q38fn-p3-mamba-only-v9`) additionally carries a single-file overlay layer
whose Dockerfile is not in the tree — the *content* is reproducible from this branch (verified
by md5 against the running container, 3,523/3,523 files), the *build* is not.

## Vendor sync

`pennyroyal` upstream moved to 3.1.x (+160 commits at time of writing). The newer Mamba/host work
there is on paths this deployment does not run (NIXL host pool `812c11b754`, DFlash
`e7a57a2afb`, `d817ab6d51`); the Mamba-backup ownership family is already merged here.

---

<a id="upstream-penny-royal-readme-unmodified"></a>

## Upstream Penny Royal README (unmodified)

Everything below is the original `README.md` from `jpezzulli/sglang-rtxpro6000` at the revision
this fork is based on, kept verbatim.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/penny-royal/logos/logo-horizontal-dark.svg">
    <img src="assets/brand/penny-royal/logos/logo-horizontal-light.svg" alt="Penny Royal" width="480">
  </picture>
</p>

# Penny Royal — Qwen3.8 on RTX PRO 6000

Run long conversations, coding agents, and tool-heavy workflows on your own
hardware. Penny Royal is an SGLang-derived runtime tuned for **NVIDIA RTX PRO
6000 Blackwell (SM120), with one card (TP1) or two (TP2)**.

**Persistent conversation caching is at the heart of Penny.** HiCache and NIXL
keep reusable state in system RAM and on NVMe, beyond what fits on the GPU,
so long-running sessions can pick up from cached history.

**524K context · HiCache + NIXL · FR-Spec · FlashInfer verification · DFlash2**

[Get started](#build-and-launch) · [Conversation cache](#hicache-and-nixl) ·
[Performance](#performance) ·
[3.1.6 release notes](RELEASE_NOTES_3.1.6.md) · [Community results](COMMUNITY-RESULTS.md)

**Penny Royal 3.1:** Flash-Next runs native NEXTN MTP with FR-Spec as the
maintained default profile, with automatic KV pool sizing replacing the preset
cap; Flash-Next DFlash2 with a separate draft model, fixed concurrency
width routing (block 6 for one request, block 4 for two or more) and
FlashInfer draft-token verification is an **experimental / alpha** profile,
fully available to opt into.

**3.1.6 (published 2026-11-06):** faster prefill on long cached prefixes, optional shared image preprocessing, and a hybrid-cache prefetch fix. The verified container is `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.6`, promoted without rebuilding. See the [3.1.6 notes](RELEASE_NOTES_3.1.6.md).

**3.1.5 (published 2026-11-03):** bounds temporary GPU memory in hybrid-Mamba checkpoint copies. The release image is `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.5`, promoted without rebuilding; see the [3.1.5 notes](RELEASE_NOTES_3.1.5.md). [3.1.4](RELEASE_NOTES_3.1.4.md), published 2026-10-29, stays available as the previous release.

**3.1.4 (published 2026-10-29):** fixes Anthropic `tool_reference` histories, improves Responses tools, replay and image handling, and installs the dependency versions qualified on the native host. [Release notes](RELEASE_NOTES_3.1.4.md). [3.1.3](RELEASE_NOTES_3.1.3.md) stays available as the previous release.

<a id="dflash2-maturity"></a>

> **DFlash2:** 27B is established and qualified; Flash-Next DFlash2 is alpha. Both need extra host RAM for the draft cache — **24.83 GB extra** in our 27B run. [RAM guidance](CONFIGURATION.md#capacity-and-memory).

<a id="hicache-and-nixl"></a>

## Built for long conversations and agents

Every tool call adds another turn. A coding session can build up hundreds of
thousands of tokens of history, and processing that history again costs time.

Penny keeps reusable conversation state across three layers:

| Where | What it does |
|---|---|
| **GPU memory** | Keeps active conversations ready for the next turn. |
| **System RAM — HiCache** | Holds reusable state beyond what fits on the GPU and brings it back when needed. |
| **NVMe storage — NIXL** | Keeps cached state on disk so matching conversation history can be restored, including after a server restart with the same configuration. |

When your client sends the next turn, Penny reuses the matching cached history
and processes the new material.

Qwen's hybrid models need their recurrent state restored alongside the
attention cache. Penny handles that complete state, including Flash-Next's
additional PLE and QSA state, so restored conversations can continue correctly.

Penny restored a **490K-token conversation** from cache and processed just
**five new tokens**, skipping almost half a million tokens of repeated prefill.

Both model profiles use this hierarchy. **NIXL disk caching is on by default**;
you can disable it while keeping GPU and RAM caching. The
[run guide](RUN.md#choose-hicache-ram-size) covers sizing and setup, and
[memory and persistence](MEMORY-AND-PERSISTENCE.md#hicache-and-nixl) explains
how the state is stored and restored.

## What's new in 3.1

Flash-Next keeps **native NEXTN MTP with FR-Spec as the maintained default**,
and its verify width is now **fixed by active concurrency** - block 8 for a
single running request, block 4 for two or more - so the width is identical at
every boot. **DFlash2 with FR-Spec** and its own draft model,
`incoai/Qwen3.8-Flash-Next-DFlash2`, is an **experimental / alpha** opt-in
profile with the same concurrency-routed widths (block 6 at one request, block
4 at two or more), and it now verifies accepted draft tokens with
**FlashInfer's SM120 kernel**.
The **KV pool is sized automatically** from profiled memory instead of a preset
cap; an explicit `MAX_TOTAL_TOKENS` remains available. Scheduler admission,
stop-regex inside speculative blocks and mixed-probability requests are fixed.
See the
[3.1 release notes](RELEASE_NOTES_3.1.md) for the full change list and the
[configuration reference](CONFIGURATION.md) for every supported setting.

<a id="models-and-launch-recipes"></a>
<a id="qualified-model-profiles-and-launch-recipes"></a>

## Two model profiles

Both profiles support **524,288-token context**, tool calling, reasoning,
images, HiCache RAM prefix caching, and optional NIXL disk caching.

| Model | Weights | Speculative decoding | Checkpoints |
|---|---|---|---|
| **Qwen3.8 Flash-Next** | NVFP4 experts, automatic FP8 dense layers on SM120 | Maintained native NEXTN MTP with FR-Spec is the default, width routed by batch size (block 8 at C1, block 4 at C2+); **experimental / alpha** DFlash2 opt-in profile (separate draft model, FR-Spec, block 6 at C1 / block 4 at C2+) verifies draft tokens with FlashInfer | [Our OrcaRouter ModelOpt conversion](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) · [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) · [Flash-Next DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) |
| **Qwen3.8-27B** | FP8 | DFlash2 (established and qualified integration), with a separate draft model at fixed block 8 | [Qwen FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) · [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) |

I personally run Flash-Next on **one RTX PRO 6000 at C=6**, with **524,288-token
context** and the KV pool sized automatically from profiled memory; the final
3.1 measurements resolved a shared **target KV pool** of **1,081,792 tokens**
on the maintained native-MTP profile and **1,210,944 tokens** on the optional
DFlash2 profile of this system. The profile reports the resolved number at
startup.
That pool is shared across active requests and cached prefixes.

### How draft width is routed

Both Flash-Next speculation paths read their verification width from a bundled
fixed policy, keyed only by how many requests are generating together: the
maintained native-MTP recipes from `configs/pennyroyal/adaptive-next.json`,
and the optional DFlash2 profile (NVFP4 target with the separate
`incoai/Qwen3.8-Flash-Next-DFlash2` draft and FR-Spec) from
`configs/pennyroyal/adaptive-dflash2.json`:

| Requests generating together | Native MTP (`next`, `next-plain`) | DFlash2 (`next-dflash2`, optional) |
|---|---|---|
| **C=1** | Fixed eight-token blocks. | Fixed six-token blocks. |
| **C=2 or more** | Fixed four-token blocks. | Fixed four-token blocks. |

One candidate width per tier means the width is a lookup, so it is identical
between boots, and a single request gets the wider draft without making every
request in a busy batch pay that extra draft work. On the maintained native-MTP
recipes that policy is a choice — `PENNY_MTP_POLICY=tiered` (the table above),
`adaptive` (the v3.0.2 policy, which may move between four and eight tokens at
one request) or `fixed-w4` (four tokens at every concurrency) — and it is one
setting on every launch route: `./configure-penny` saves it, a launch folder
carries it in its startup script, and
`PENNY_MTP_POLICY=fixed-w4 ./run-penny` or the same variable in front of
`docker compose up` decides it while nothing is saved
([details](CONFIGURATION.md#speculation-width-policy)). **C is request concurrency;
TP is the number of inference GPUs.**

**FR-Spec rides along at every concurrency.** FR-Spec narrows the vocabulary
searched by the draft head; routing selects how many tokens it drafts. The
target model keeps its full vocabulary. The maintained default is the
[FR-Spec native-MTP recipe](RUN.md#launch-flash-next-with-fr-spec), with the
[non-FR recipe](RUN.md#launch-flash-next-without-fr-spec-alternative) also
selectable - without the map, and with the same block 8 / block 4 routing; the
experimental / alpha [DFlash2 recipe](RUN.md#launch-flash-next-with-dflash2) -
which verifies its draft tokens with FlashInfer's SM120 kernel - remains
available as an opt-in. DFlash2 offers more token-pool capacity; we are still
evaluating performance across concurrency levels. TP2 for the DFlash2
profile is expected to work; it has not been tested on two-GPU hardware.

For 27B alternatives, see the [Swift 27B results](SWIFT-27B.md) and the
[OrcaRouter FP8 checkpoint](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8).

<a id="performance-and-qualification"></a>
<a id="performance-and-testing"></a>
<a id="flash-next-v250-online-fp8-performance-at-a-glance"></a>

## Performance

### Flash-Next 3.1

**Native MTP default: 252.83 tok/s mean / 501.63 peak on a real coding
workflow, and 216.23 / 525.80 on the ten-slide presentation (agentic), single
request at block 8.**

Measured on one RTX PRO 6000 Blackwell, using our OrcaRouter ModelOpt NVFP4
checkpoint, FR-Spec, a 524K context limit, and normal non-greedy sampling. The
native rows are the final October 21 screens on the merged source
(`fe5f02c`); the DFlash2 rows keep their `a60f442` labels. Every rate is
SGLang's native generation rate, including reasoning; prefill, tool execution,
and idle time are separate.

| Profile / workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Native MTP (maintained default), single request, C1 W8 | **197.87** | **345.30** |
| Native MTP, full four-request batch, aggregate, C4 W4 | **555.11** | **731.06** |
| Native MTP, full six-request batch, aggregate, C6 W4 | **472.79** | **528.49** |
| Native MTP, coding workflow (agentic), C1 W8 | **252.83** | **501.63** |
| Native MTP, presentation workflow (agentic), C1 W8 | **216.23** | **525.80** |
| DFlash2 + FlashInfer (optional profile), single request, C1 W6 | **241.49** | **339.53** |
| DFlash2 + FlashInfer, six concurrent requests, aggregate, C6 W4 | **504.80** | **568.90** |
| DFlash2 + FlashInfer, presentation workflow (agentic), C1 W6 | **217.19** | **448.70** |

Both native workflows completed end to end at block 8 with one request
generating, and the finished presentation deck previewed.
[Current results](RESULTS.md#penny-royal-31) records the sources, intervals
and settings behind every number, plus the dated earlier screens.

| Cold prompt (Flash-Next DFlash2) | Mean prefill tok/s |
|---|---:|
| 63,906 tokens | **15,365.38** |
| 489,921 tokens | **8,715.05** |

The Flash-Next DFlash2 reasoning qualification scored **93.46/100**. See
[results and measurement settings](RESULTS.md#penny-royal-31)
for the tool checks, cache restore, sampling details, and earlier results.

[RESULTS.md](RESULTS.md#penny-royal-313-october-27-2026) adds the dated decode
and time-to-first-token rows measured on the 3.1.3 candidate on both qualified
profiles, with the launched configuration and interval counts. The Flash-Next
rows were measured with `adaptive` selected locally while the public width-policy
default stays `tiered`, so they are not a matched-baseline comparison and carry
no speed-up claim.

<a id="27b-fp8--dflash2-performance-at-a-glance"></a>

### Qwen3.8-27B FP8 with DFlash2

Current recorded performance on one RTX PRO 6000, fixed block 8:

| Workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Single request | **133.06** | **232.07** |
| Four concurrent requests, aggregate | **459.74** | **570.73** |

| Cold prompt | Mean prefill tok/s |
|---|---:|
| 63,906 tokens | 6,421.59 |
| 489,921 tokens | 1,677.10 |

The reasoning suite scored **96.94/100** on this profile.

[Benchmark history](RESULTS.md) keeps the earlier records — the October 10 3.0
screens, the 2.5.3 comparison, the public code-edit benchmark, and the dated
27B campaigns — with their original labels and timing methods.

<a id="independent-tp2-fp8-validation"></a>

Running two cards? [Community results](COMMUNITY-RESULTS.md) covers TP2 setups
and measurements from other users.

## Build and launch

Choose the installation path that suits your machine:

| Path | What you do |
|---|---|
| **[Prebuilt container](docker/pennyroyal/README.md)** | Download the image and launch files, edit your host configuration, and run it. Models and caches stay on the host. |
| **[Native installation](BUILD.md)** | Build the runtime, set your model and cache paths, and use the [native startup recipe](RUN.md). |

Both paths work with ordinary configuration files. The optional
**[beta configurator](CONFIGURE.md)** helps write those files and prints the
launch command. It is open for testing; please [report issues](https://github.com/jpezzulli/sglang-rtxpro6000/issues)
if you try it. The complete list of supported settings, defaults, interactions
and restart/cache effects is the [configuration reference](CONFIGURATION.md).

### Your container configuration stays outside the image

```text
your-host/
  run.sh                              image, GPU selection, ports and mounts
  config/start-flash-next-frspec.sh   model and SGLang startup settings
  config/nixl-posix-frspec.toml        optional disk-cache settings
```

Prepare those files before starting. `run.sh` starts Docker and mounts your
configuration directory read-only at `/config`; the container runs the startup
file from that mount. Models and writable caches are mounted separately.
Edit the host files and restart to apply changes. You can download a newer
image while keeping your own configuration—no container rebuild required.

The [container guide](docker/pennyroyal/README.md) provides the files and
commands; the beta configurator can generate the same ordinary files for you.

### Hardware and memory

The inference target is **RTX PRO 6000 Blackwell, TP1 or TP2**. An Ampere or
Ada Lovelace GPU can be used separately for image preprocessing; it is not an
extra tensor-parallel inference card. CPU preprocessing is also available. The
build is narrowed to RTX Blackwell inference and Ampere/Ada sidecar image
processing — other GPU build targets are trimmed, which keeps the release
slimmer and easier to download.

| Profile | Default host cache | Additional model-related host memory |
|---|---|---|
| Flash-Next | 32 GB HiCache | About 48 GiB for RAM-backed PLE |
| Flash-Next/DFlash2 | 32 GB HiCache | The draft's own host cache, on top of the target's budget |
| 27B/DFlash2 | 96 GB HiCache | Draft cache plus loading and runtime allocations |

A DFlash2 draft keeps a host cache of its own that is **additional to** the
HiCache figure above, so size host RAM from what a boot reports rather than
from the HiCache setting alone. The recorded 27B launch spent 79.45 GB of
target KV plus 16.63 GB of Mamba state - the 96 GB target/Mamba budget - and a
further 24.83 GB of draft sliding-window cache, about 120.91 GB of caches
before other process memory. This release documents that bill; it does not
reduce it. See [limitations](LIMITATIONS.md) and the
[configuration reference](CONFIGURATION.md#capacity-and-memory).

Leave room for the operating system and model loading. You can
[change the RAM cache size](RUN.md#choose-hicache-ram-size), or move Flash-Next's
PLE table to an SSD with [NVMe PLE](NVME-PLE.md).

NIXL disk caching is **on by default and optional**. Switching it off keeps GPU
and RAM prefix caching. NVMe PLE is a separate choice. The [launch guide](RUN.md)
explains cache sizing, GPU selection, concurrency, and startup.

The first start compiles kernels and captures CUDA graphs. Wait for the
server-ready message, then [check the API](RUN.md#smoke-through-the-normal-api).

**Help shape Penny Royal’s optimization priorities.** [Share your deployment and
workload](https://github.com/jpezzulli/sglang-rtxpro6000/discussions/154). Your
hardware, model, typical context size, concurrency, and biggest bottleneck help
guide future performance, memory, serving, and compatibility work.

## Built for the way we use it

Penny Royal started as my home runtime on thegrid. I wanted long conversations
and useful coding agents on the hardware I already owned. The tuning follows
that workload: keep several requests moving, make tool calls usable, and avoid
reprocessing a long conversation when its state can be restored.

The name comes from Penny Royal in Neal Asher's Polity books.

Thanks to **[aiueo52](https://github.com/aiueo52)** for the Flash-Next optimization
work and discussion, **[LandOfLemons](https://github.com/LandOfLemons)** for the
session-cache and WSL2 contributions, **[cube4elements](https://github.com/cube4elements)**
for the image-request crash fix, and **[untcoder2](https://github.com/untcoder2)**,
**[StockSpecialist1707](https://www.reddit.com/user/StockSpecialist1707/)**, and
**[palves](https://github.com/palves)** for field testing and setup feedback.
For 3.1, **[@jhsmith409](https://github.com/jhsmith409)** and
**[@LocalhostRonin](https://github.com/LocalhostRonin)** contributed directly.
[Penny Royal](mailto:Pennyroyal@agentmail.to) did almost all the coding across
these releases, including the ports and integration work. [John](https://github.com/jpezzulli),
Penny and Codex worked through the architecture, implementation, tuning,
verification-backend choice and debugging together, with plenty of back and forth
over what was actually worth keeping.

## Documentation

| I want to… | Read |
|---|---|
| Install or upgrade | [Native build](BUILD.md) · [Container](docker/pennyroyal/README.md) |
| Choose launch, cache, GPU, and context settings | [Run guide](RUN.md) · [Configuration reference](CONFIGURATION.md) |
| Understand conversation caching | [HiCache and NIXL](#hicache-and-nixl) · [Memory and persistence](MEMORY-AND-PERSISTENCE.md#hicache-and-nixl) |
| Try guided setup | [Beta configurator](CONFIGURE.md) |
| Understand the FP8 and SSD options | [Online FP8](FP8.md) · [NVMe PLE](NVME-PLE.md) |
| See measurements | [Results and history](RESULTS.md) · [Community reports](COMMUNITY-RESULTS.md) |
| Read the implementation details | [Backends](BACKENDS.md) · [Memory and persistence](MEMORY-AND-PERSISTENCE.md) |
| Check changes and source lineage | [3.1.6 notes](RELEASE_NOTES_3.1.6.md) · [Machine-readable notes](release-3.1.6.json) · [History](CHANGES.md) · [Provenance](PROVENANCE.md) |
| Check known issues | [Limitations](LIMITATIONS.md) |
| Use an AI assistant for setup | [llms.txt](llms.txt) |

Our [validation suite](https://github.com/jpezzulli/pennyroyal-validation)
contains the reasoning, tool, image, context, and performance tests.

To hear about new releases, choose **Watch → Custom → Releases** on GitHub.
If you build on Penny Royal, please link this repository and the version you
started from. Feedback, contributions, and real workload reports are welcome.

<p align="center">
  <img src="assets/brand/penny-royal/artwork/penny-royal-crystal-transparent.png" alt="Penny Royal crystal artwork" width="240">
</p>

## License

The SGLang-derived source is Apache-2.0. Model checkpoints and third-party
components retain their own licenses.

<!-- Preserve existing README links while detailed material lives on its own page. -->
<a id="current-release--pennyroyal-v251"></a>
<a id="current-release--pennyroyal-v250"></a>
<a id="current-release--pennyroyal-v253"></a>
<a id="v250-options--online-fp8-and-nvme-ple"></a>
<a id="retained-v240-flash-next-prefill-measurements"></a>
<a id="qwen38-27b-fp8dflash2--august-24-2026-campaign"></a>
<a id="27b-real-agentic-context-behavior"></a>
<a id="earlier-releases"></a>
<a id="v23--faster-flash-next-decoding-with-fr-spec"></a>
<a id="previous-v212--maintenance-release"></a>
<a id="qualified-configuration-matrix"></a>
<a id="resolved-backends-qwen38-27bdflash2"></a>
<a id="resolved-backends-qwen38-flash-next"></a>
<a id="sm120-enablement-was-scoped-not-indiscriminate"></a>
<a id="how-pennyroyal-selects-sm120-paths"></a>
<a id="memory-recovery-and-automatic-kv-sizing"></a>
<a id="scope-and-limitations"></a>
<a id="documentation-map"></a>

[3.1.6 release notes](RELEASE_NOTES_3.1.6.md) · [Benchmark history](RESULTS.md) ·
[Configuration reference](CONFIGURATION.md) · [Known issues](LIMITATIONS.md)

<!-- Compatibility anchors for earlier versions of this guide. -->
<a id="broader-model-support-without-hicachenixl"></a>
<a id="building-on-pennyroyal"></a>
<a id="choosing-your-settings"></a>
<a id="flash-next-precision-and-placement"></a>
<a id="hardware-storage-and-first-start-expectations"></a>
<a id="hicachenixl-persistence"></a>
<a id="model-profiles-and-launch-recipes"></a>
<a id="prebuilt-docker-image-for-qwen38-on-rtx-pro-6000"></a>
<a id="qwen38-flash-next-nvfp4-with-online-fp8"></a>
<a id="sglang-for-qwen38-on-one-nvidia-rtx-pro-6000-blackwell-sm120"></a>
<a id="source-changes-and-upstream-work"></a>
<a id="technical-reference"></a>
<a id="validation-suite"></a>
<a id="why-this-runtime-exists"></a>
