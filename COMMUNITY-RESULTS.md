# Community results

Pennyroyal is a home-runtime project shared publicly. These reports show how
other people are using it: sustained agent work, different GPU variants, both
model profiles, and multi-GPU deployments. Thank you to everyone who shared
their configurations, measurements, and problems.

**These are independent users' results, not our measurements. We have not
independently reproduced them.** Hardware, versions, workloads, and timing
methods differ. Each report keeps its own context; this is not a leaderboard
or a promise of the same performance on another system. Our own dated tests
remain in [RESULTS.md](RESULTS.md).

- [kazimirek: more than a week of agentic work on a 300 W Max-Q](#kazimirek--a-week-of-agentic-work-on-max-q)
- [WonderRico: both profiles on GSM8K and Automation Bench](#wonderrico--both-profiles-on-gsm8k-and-automation-bench)
- [StockSpecialist1707: a larger single-GPU pool and online FP8](#stockspecialist1707--single-gpu-capacity-and-online-fp8)
- [H3PO: two-GPU Flash-Next FP8 and NVFP4](#h3po--two-gpu-flash-next)

## kazimirek — a week of agentic work on Max-Q

**Reported September 12, 2026** by
[u/kazimirek](https://www.reddit.com/user/kazimirek/).
[Original field report](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9cn6h6/).

> Your runtime has been running a real workload for me for over a week now and it just works

**4.8 billion prompt tokens with a reported 96% prefix-cache hit rate**, alongside
**55 million generated tokens**, is the headline from this week-long workload.
The author used Claude Code through LiteLLM: one orchestrator and up to three
subagents, contexts usually **150K–236K**, with images in the mix.

The cache reuse matters as much as the decode speed: long agent conversations
repeatedly carry earlier context forward. Reusing that work avoids processing
the entire prefix from scratch on each turn. These are reported operational
totals, not 4.8 billion tokens of fresh prefill computation or a measured
cache-on/cache-off speedup.

### Reported configuration

| Item | Configuration |
|---|---|
| Hardware | One RTX PRO 6000 Blackwell Max-Q, 300 W variant; 125 GB system RAM; TP1 |
| Runtime | Pennyroyal v2.1.1, source `fb1216c6c` |
| Target | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Context and GPU KV | 262K context; approximately 524K GPU KV tokens; FP8 E4M3 KV |
| Host cache | 20 GB HiCache tier, reported as approximately 1.06M tokens |
| Concurrency and state | Four running requests; 40 Mamba slots |
| Prefill and speculation | 4K chunks; NEXTN MTP, three steps / four draft tokens |
| Other settings | PIL image preprocessing; expandable segments |

### Reported performance over the run

These are **medians of scheduler-log throughput samples**, grouped by running
request count. They are not medians of completed-request latency or our
synchronized fixed-output benchmark.

| Running requests | Reported median throughput |
|---|---:|
| 1 | 118 tok/s |
| 2 | 207 tok/s aggregate |
| 3 | 272 tok/s aggregate |
| 4 | 326 tok/s aggregate |

The author also reported C1 mean 125 tok/s and p90 174; C4 p90 379 tok/s;
approximately 200 tok/s aggregate overall, with peaks around 520; and MTP
acceptance around 0.42. Prefill was reported at approximately **9.8K tok/s per
chunk while decoding concurrently**, and **16K tok/s when not decoding**.
Those prefill figures are not a specified-length cold-prefill test.

**Operational notes:** the author described reliable ongoing use and reported
no server errors in the workload summary, but also described two issues they
had encountered: streaming tool-call corruption and image-preprocessing OOM
when GPU memory was full. They pointed to later maintenance work; our
[changelog](CHANGES.md) records the actual fixes and release history. We do
not turn the report into a claim of an incident-free week. The complete log,
counter-collection method, and incident timing were not supplied in the comment.
This report predates their planned v2.5 test and is not Max-Q qualification of
v2.5's online FP8 option.

## WonderRico — both profiles on GSM8K and Automation Bench

**Reported September 12, 2026** by
[u/WonderRico](https://www.reddit.com/user/WonderRico/), using the updated
Pennyroyal repository. The comments include repeated scheduler-log samples
for **27B FP8 with FP8 KV**, then **Flash-Next NVFP4 with FP8 KV and
`SGLANG_SM120_ONLINE_MXFP8` enabled**.

| Model and workload | Author's throughput summary | Range of posted C4 samples | Posted C4 acceptance length |
|---|---|---:|---:|
| [27B FP8, GSM8K](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9cv7a2/) | Up to about 800 tok/s sustained | 743–819 tok/s aggregate | 6.11–6.49 |
| [27B FP8, Automation Bench](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9cwdf9/) | About 500–600 tok/s | 491–670 tok/s aggregate | 4.95–5.82 |
| [Flash-Next, GSM8K](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9esqd5/) | About 480–580 tok/s sustained | 476–557 tok/s aggregate | 3.22–3.40 |
| [Flash-Next, Automation Bench](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9ewf4k/) | About 400–500 tok/s | 399–494 tok/s aggregate | 2.75–3.18 |

The ranges above are rounded from the posted lines with **exactly four running
requests**. Lines with three running requests are not included in those C4
ranges. The author's whole-run summary and the range of a posted excerpt are
different observations, which is why both are shown.

GSM8K used very short, one-turn prompts. Automation Bench was described as
more agentic; its log excerpts show substantially larger active token counts.
Those counts are batch totals, not per-request context lengths. The samples
show CUDA graphs active and queued work keeping the server busy. Scheduler
throughput windows are not whole-request or synchronized-batch measurements.

**Scope:** these comments do not specify an exact target repository for 27B,
source SHA, full launcher, GPU model/count, power setting, or host RAM. We do
not infer those from other users' reports. No benchmark accuracy scores or
full-run exports were included, so these logs establish reported throughput
and acceptance, not reasoning/tool-quality parity or a direct comparison with
our longer-context tests. In a separate
[configuration discussion](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9c961f/),
the author preferred RAM PLE because SSD PLE's decode tradeoff was too large
for their use; no matched numeric comparison was supplied there.

## StockSpecialist1707 — single-GPU capacity and online FP8

**Reported September 12, 2026** by
[u/StockSpecialist1707](https://www.reddit.com/user/StockSpecialist1707/).
[Original report](https://www.reddit.com/r/BlackwellPerformance/comments/1weaxdz/comment/p9e3xao/).

The author used the v2.5.0 Flash-Next FR-Spec recipe with online FP8 on a
**single RTX PRO 6000 Blackwell Workstation Edition, 96 GB, TP1**, driver
595.84 and CUDA 13.2, with no user-imposed GPU power cap. Their system has two
cards, but the model was pinned to one by UUID and the second was idle for
these measurements. **This is not a TP2 result.** The full checkpoint ID and
host RAM were not stated in the comment; other settings were described as stock.

### Pool capacity

| `MAX_TOTAL_TOKENS` setting | Reported resulting pool | Reported `available_gpu_mem` | Reported 1,024-token test wall time, median of three |
|---|---:|---:|---:|
| Unset — recipe default | 824,384 | 3.66 GB | 3.86 s |
| 1,048,576 | 1,048,576 | 4.67 GB | Not reported |
| 1,310,720 | 1,114,304, clamped by server | 3.81 GB | 3.79 s |

The reported larger pool is approximately **35% above 824,384**. Served context
remained **524,288 with factor-2 YaRN**. Pool capacity is shared storage, not a
larger per-request context window or a test of filling that entire pool.
The author had not tested needles above 490K on this configuration.

**Recipe clarification:** the original comment calls the unset row automatic
sizing. In the published v2.5.0 FR-Spec recipe, unset `MAX_TOTAL_TOKENS`
actually selects the explicit **824,384 default cap**. The report's pool and
free-memory numbers are preserved, but they do not by themselves establish an
estimator defect. A clamped startup allocation also does not guarantee enough
transient memory for every later workload.

### Reported optimization comparison

| Configuration label from the report | Test wall time, median of three | Reported mean server decode |
|---|---:|---:|
| v2.5 plain | 4.91 s | 212 tok/s |
| + FR-Spec | 4.44 s | 225 tok/s |
| + online FP8 | 3.86 s | 272 tok/s |

The same prompt was used for three runs of each configuration; the report also
mentions a 360 tok/s peak. These wall times and server means are not our
post-first-token metric. Complete request payloads and server usage were not
provided, so we do not derive an additional token rate from the wall times or
claim a statistically established zero decode cost for the larger pool.

The author identifies the system as BESTIA, managed with Prometeus, and
discloses that Claude drafted the write-up from their logs. They offered a
future two-card sweep; no result from that proposed sweep is included here.

## H3PO — two-GPU Flash-Next

**Reported August 28, 2026** by
[u/H3PO](https://www.reddit.com/user/H3PO/), on an earlier Pennyroyal branch,
before v2.5.0; the exact source SHA was not stated in the benchmark comments.
The deployment used **two RTX PRO 6000s, TP2/EP2, without NVLink**, FP8 KV,
and native NEXTN MTP. It was a Docker deployment built from the branch, not
evidence that the repository's inherited Dockerfile reproduces our native
qualified setup.

### FP8 target

The [FP8 report](https://www.reddit.com/r/BlackwellPerformance/comments/1w04xb7/comment/p6ek6e8/)
used `Qwen/Qwen3.8-Flash-Next-FP8` and a StackOverflow-derived coding corpus.
The `llama-benchy` invocation used 4,096 prompt tokens, 1,024 generated tokens,
generation-latency mode and three runs at each concurrency. Reported
`mem-fraction-static=0.95` produced a 3,182,848-token KV pool. The shared
configuration used 524,288 context; the much larger pool is not context proof.

| Workload | Reported total throughput |
|---|---:|
| C1 prefill, 4,096 tokens | 13,414.47 ± 82.74 tok/s |
| C1 decode | 200.93 ± 12.87 tok/s |
| C2 decode | 295.88 ± 29.07 tok/s aggregate |
| C4 decode | 330.76 ± 22.04 tok/s aggregate |
| C6 decode | 339.34 ± 11.14 tok/s aggregate |

The original wide table, including per-request rates, peaks and high-variance
concurrent-prefill rows, remains in the
[retained TP2 result](RESULTS.md#independent-tp2-fp8-validation). The `±`
notation is preserved as supplied; these are not relabeled as our medians.

### NVFP4 target follow-up

H3PO subsequently reported the same benchmark against
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, with FP8 KV and MTP, cards at **600 W**,
and `mem-fraction-static=0.90`. The author reported reducing the memory
fraction after a Mamba runtime OOM on multimodal requests with the earlier
configuration, and an available KV capacity of 5,787,136 tokens.
[Original follow-up](https://www.reddit.com/r/BlackwellPerformance/comments/1w04xb7/comment/p6go1j3/).

| Workload | Reported total throughput |
|---|---:|
| C1 prefill, 4,096 tokens | 13,576.99 ± 819.88 tok/s |
| C1 decode | 204.10 ± 10.60 tok/s |
| C2 decode | 338.99 ± 5.68 tok/s aggregate |
| C4 decode | 386.74 ± 41.13 tok/s aggregate |
| C8 decode | 427.46 ± 24.02 tok/s aggregate |
| C11 decode | 459.94 ± 13.08 tok/s aggregate |

**Configuration lessons and limits:** removing the optional
`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` setting allowed native MTP to work.
A separate custom-allreduce graph-capture failure had been resolved by
removing `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from this TP2 setup.
The [diagnostic exchange](https://www.reddit.com/r/BlackwellPerformance/comments/1w04xb7/comment/p6dtvoh/)
keeps those two problems distinct. Do not generalize that allocator setting
to our qualified TP1 recipe. Neither table supplies MTP acceptance or a
full reasoning/tool/media qualification; high-concurrency prefill was noisy.
This is useful independent multi-GPU evidence, not a controlled comparison
with our TP1 results or a claim that all TP2/TP4 combinations have been tested.

## Share a field report

Reports of ordinary use are welcome alongside benchmarks. Include your
Pennyroyal version/source, target and draft models, hardware and power settings,
context and pool sizes, concurrency, relevant launcher changes, and how you
measured the result. Say whether rates are per request or aggregate, and
whether they are medians, peaks, scheduler samples, or completed-request rates.
Include problems and tradeoffs as well as successes. Remove credentials,
private prompts and personal data before sharing logs through
[GitHub issues](https://github.com/jpezzulli/sglang-rtxpro6000/issues) or a public
discussion. A public report link and your preferred attribution make it easier
to credit your work accurately.
