# External optimization research for four V100 GPUs

Checked on 13 November 2026. The next useful work is to measure expert transfer cost, graph reuse, and long-context QSA work. Larger speculative windows and more prefetching need evidence that they reduce the full request time. This applies to four 16 GB V100 GPUs on two CPU sockets, without NVLink. One GPU has a PCIe Gen3 x8 link. The model is the earlier Unsloth UD-IQ4_XS Qwen3.8-Flash-Next set.

The [initial deployment audit](v100-four-gpu-research-2026-11-13.md) contains the hardware and transfer measurements. Its deployment table records the state before the upstream integration. This review adds sources outside Strata. Dates below are publication or revision dates when the source gives one. A page checked today is not necessarily published today.

## Recent papers and their deployment limits

| Source and date | Finding | Experiment for this machine |
| --- | --- | --- |
| [Bevy](https://arxiv.org/html/2611.04794v1), 3 November 2026 | Ready requests can advance to later layers while other requests wait for experts. The scheduler groups ready jobs at the same layer and gives lagging requests priority. It shares pending expert reads and holds cache entries until GPU work finishes. | First measure idle gaps with two clients. Consider a small scheduler experiment only if expert waits remain large. Keep per-request recurrent state and cache ownership separate. |
| [LayerScope v2](https://arxiv.org/abs/2509.23638v2), 16 April 2026 | This is the newer title of the PreScope paper. It combines layer-specific prediction, cross-layer scheduling, and asynchronous I/O. | Measure prediction precision on **cache misses**, wasted bytes, and useful overlap. Use a trained predictor only if it beats the existing demand path after its compute and memory costs. |
| [DraftExpert](https://arxiv.org/abs/2607.24434), 27 July 2026 | A longer verification block can activate more target experts. A trained small draft expert and an expansion-aware stopping rule limit that cost. Final tokens still pass target verification. | Log unique experts and transferred bytes per accepted token at speculation depths 2 and 4. An adaptive stopping rule is a smaller first experiment than training a new drafter. |
| [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/), 24 March 2026 | Rotation and vector quantization can reduce KV storage. The published attention speed examples use H100. | Compare local TQ6/TQ8 with INT8 at long context. Include quantize, gather, attention, snapshot, and restore time. Retain the current INT8 service profile until this comparison passes. |
| [OmniKVQuant](https://arxiv.org/abs/2609.11582), 10 September 2026 | On Qwen Omni models, rotation-based cache quantization encounters temporal key drift and different value geometry across modalities. | Include image-conditioned continuations in any KV-format change. These Omni results do not establish a defect in Flash-Next or its CPU image encoder. |

Bevy is the newest directly relevant scheduling paper found in this review. Its test platforms are an M6 Mac mini and Jetson AGX Thor. Experts come from SSD. Its concurrency sweep uses 30% expert residency and 2–16 requests. With two requests, the gain over the fastest baseline can be as small as 1.01 times. Its largest headline gain is therefore not a forecast for this RAM-resident, two-client V100 service. The useful idea is scheduling by actual readiness. [Bevy evaluation](https://arxiv.org/html/2611.04794v1#S5).

DraftExpert's reported 1.45 times average gain uses DeepSeek-V2-Lite and Moonlight-16B-A3B. It needs a trained drafter. Its expansion metric is a design reference for the existing MTP path, not a compatible model file. [DraftExpert abstract](https://arxiv.org/abs/2607.24434).

## V100 implementations and current discussions

| Source | Evidence and useful detail | Local implication |
| --- | --- | --- |
| [llama.cpp PR 29753](https://github.com/ggml-org/llama.cpp/pull/29753), merged 1 October 2026 | Routing SM70 K-quant batch-one MMVQ to a two-warp table improved dense Qwen3.8-27B Q4_K_M decode by about 3.2% on one 32 GB V100. The author used six interleaved comparison blocks and equal perplexity. | Inspect the actual IQ3_S/IQ4_NL dispatch before tuning warp count. This patch changes llama.cpp MMVQ; it does not directly patch Strata's grouped expert kernels. |
| [llama.cpp issue 20605](https://github.com/ggml-org/llama.cpp/issues/20605), 15 March 2026, closed | A user reports a V100 regression when pipeline-parallel graph reuse was disabled. The test used two 32 GB V100 GPUs with NVLink and Qwen3.5. | Count graph construction, capture, instantiation, eviction, and replay by batch layout. Strata already caches several graph forms; adding another cache without this trace can duplicate existing work. |
| [anyei V100 fork](https://github.com/anyei/llamacpp-v100/blob/master/README.md), dated results through 1 October 2026 | A host executor runs expert misses while GPU hits execute. Later graph reuse made CUDA graphs beneficial again. A separate block-top-k change altered selection at the cutoff. | Examine CPU/GPU overlap and graph lifetime together. Keep exact mask skipping separate from selector changes that alter output. These are that fork's measurements. |
| [Wyvern four-V100 offload design](https://github.com/WyvernTKC/llama.cpp-4xV100/blob/sm-tensor-4xv100/docs/moe-offload.md), checked 13 November | Selected-expert copies help small batches. Large batches can touch most experts and lose to the routing readback barrier. Four staging slots reached 94% of the measured compute ceiling; twelve reached 99%, on its DeepSeek test. | Use bytes per accepted token and CPU execution time to choose transfer versus CPU work. Sweep the local ring only while accounting for lost expert-cache capacity. Its staging slots are not Strata ring slots. |
| [Flash-Next Volta fork](https://github.com/jackjusko/jusko-llama-volta-qwen3flash/blob/main/README-FORK.md), checked 13 November | Sparse masked-tile skipping helped decode. A similar prefill variant was reverted after large prompt regressions. Other gathered-QSA variants also lost at short context. | Gate sparse-attention work by phase and geometry. Test 2K and 8K as well as 64K and 128K; a faster long-context kernel can still hurt normal requests. |
| [Strata issue 5063](https://github.com/Niko1221/Strata/issues/5063), discussion through 5 November 2026 | A four-card PCIe service reports a prompt gain from `--prefill-pipe 3072`. It also reports a large regression on one newer engine and a concurrency cost from INT8 KV. | Test prompt pipeline depth independently. This host has half the VRAM, less RAM, and two clients; copy the comparison method rather than the settings. |
| [Strata issue 606](https://github.com/Niko1221/Strata/issues/606), new observation 9 November 2026 | A 61K coding prompt can produce repeated zeros on both newer and older controls. The later comment does not establish a new-release cause. | Long-context validation must check answer content, tools, and the next request after a long generation. A successful needle alone does not cover this failure. |

The initial audit also covers [three-stage V100 handoff waits](https://github.com/Niko1221/Strata/issues/5538), [layer-major prompt processing](https://github.com/Niko1221/Strata/issues/3976), and [prompt-aware stage splits](https://github.com/Niko1221/Strata/issues/5265). These remain relevant after a merge. Lower wait time can lose overall speed if draft acceptance falls. Fewer repeated reads can lose speed when the weights already fit in RAM.

## Hardware and profiler requirements

Use CUDA 12.9 to build `sm_70`. CUDA 13 removes offline compilation for Volta and several legacy CUPTI APIs. Select profiler versions and their loaded CUPTI library explicitly. A current profiler's failure does not prove that a kernel cannot be measured. Start with CUDA events and application counters, then validate the chosen tracing tool on a small program. [NVIDIA architecture support](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers), [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/).

Volta has FP16 tensor cores. Check register use, shared memory, occupancy, and the actual instruction path for its geometry. Newer GPU recipes for BF16, FP8, or asynchronous copy instructions need a separate Volta implementation. [Volta tuning guide for CUDA 12.9](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html).

Place expert pages before registration and inspect physical placement after first touch. A thread's memory policy affects future allocations and is inherited by new threads. It does not move existing file-cache pages. Keep the large PLE table in the roomier node's budget while the expert helpers allocate locally. [Linux NUMA memory policy](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html), [NVIDIA NUMA guidance](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#numa-best-practices).

Current [vLLM tuning documentation](https://docs.vllm.ai/en/stable/configuration/optimization/#numa-binding-for-multi-socket-gpu-nodes) starts each worker under a NUMA policy before Python startup. Strata uses one process across four GPUs, so one strict process-wide socket binding would be a poor translation. The transferable idea is early allocation policy plus per-stage helper affinity. This reference does not establish that current stock vLLM runs on V100.

## Prioritized experiment plan

All rows below are proposals. Use the same model, INT8 KV, two request slots, and context limit as the control unless the row explicitly tests a separate profile. Preserve tools and image support.

| Priority | Work and local entry points | Measurement and decision rule |
| --- | --- | --- |
| P1 | Attribute time and traffic. Inspect `src/core/session.cpp`, the expert source, and stage launch counters. | Record miss bytes, useful prefetch bytes, duplicate copies, CPU work, transfer wait, and GPU time per **accepted** token. Use this to select the next patch. |
| P1 | Price CPU work against the real link. GPU2 has about 6 GB/s local H2D in the audit. | Compare CPU execution with transfer plus GPU execution for each format and stage. Use a nearby split sweep around `11,24,36`; record per-card memory minima and both prompt and decode rates. |
| P1 | Tune prompt pipelines and transfer capacity. Inspect `src/prefill/prefill.cpp`. | Compare pipe off versus 3072, then prompt chunks 4096/8192. Tune ring capacity separately. Count extra memory and lost cache hits. Keep changes only if end-to-end time improves. |
| P1 | Check graph replay and eviction. Inspect `include/strata/core/verify.hpp`, `src/core/session.cpp`, and `src/core/mtp.cpp`. | Record shape keys, graph hits, capture cost, and graph bytes for one and two clients. Avoid a cache entry count increase that removes more expert capacity than it saves in launch time. |
| P2 | Test exact compact selection at long context. `STRATA_TOPK_COMPACT` already exists in `src/kernels/cuda/qsa_select.cu`. | Run off/on at 32K/64K/128K and short additions to long histories. Check selected IDs and tie behavior. Do not count this existing code as a new port. |
| P2 | Match grouped-GEMM tiles to the observed routed-row histogram. Inspect `src/kernels/cuda/iq_kernels.cu`. | Measure IQ3_S/IQ4_NL at the real small verification widths and wide prompt rows. Preserve FP64 parity gates. Require a full-service gain after the kernel gain. |
| P2 | Bound speculation by expert expansion. Inspect `src/core/mtp.cpp` and the verification path. | Compare depths 2/4 and min-p 0.5/0.7. Log accepted/offered drafts, unique experts, transfer bytes, and first output latency. Test Chinese prose, code, JSON, and tools separately. |
| P2 | Measure startup I/O separately from resident inference. Inspect native profile filling and expert-complement loading. | The compatibility runs needed several minutes to load weights; subsequent requests report no expert-file fetches. Record physical reads by input file and phase. Test read ordering and lookahead before creating a new packed layout. These functional runs are not a controlled startup-speed comparison. |
| P3 | Prototype readiness-based request scheduling from Bevy. | Start with two requests. Keep expert reference counts through GPU completion and prevent duplicate fetches. Require better aggregate throughput without excessive latency for either client or state leakage. |
| P3 | Evaluate lower-bit KV as a separate profile. | Compare TQ6/TQ8 and INT8 on content checks, long context, branches, images, memory, and total time. More free VRAM can help expert residency even when attention itself gets slower. |

First collect three fresh prompt sizes and agent continuations. Warm the adaptive expert cache before steady-state measurements, but also report cold startup separately. Interleave repeated arms where possible. Record prompt reuse and answer hashes. Include 256-token fixed-length comparisons and longer natural completions. A small decode difference on three synthetic prompts is not enough to select a default.

Keep model replacement, tensor parallelism, and per-socket replicas as separate experiments. The earlier audit already identifies alternative V100 engines and newer checkpoints. Their memory use, links, API behavior, and task quality need separate controls.

## Follow-up on 19 November 2026

The local branch merged the fork's `origin/main` and `origin/local/working`, then integrated Strata upstream through `58ee32277`. The upstream 0.2.16 source layout places CUDA code under `engine/` and the server under `terminal/serve/`. It builds Strata's own prompt kernels.

| Source | Decision |
| --- | --- |
| [jmnargi/Strata-V100 #116](https://github.com/jmnargi/Strata-V100/pull/116) and [Basalt's suffix reset](https://github.com/jesdga95/basalt/commit/d7558e4e9) | Ported the generation-stamped suffix index from jmnargi. It clears the table before a stamp is reused and preserves the hash and proposal order. Basalt's register prefetch showed a kernel gain with no repeatable model-level gain in jmnargi's tests. |
| [jmnargi's prefix matcher](https://github.com/jmnargi/Strata-V100/commit/383ad10fe) | Ported exact block-wise comparison for 32-bit token buffers in the RAM and disk cache. Mixed-width token comparisons still use the existing scalar path. |
| [jmnargi/Strata-V100 #104](https://github.com/jmnargi/Strata-V100/pull/104) and [1Cat-vLLM #1176](https://github.com/1CatAI/1Cat-vLLM/pull/1176) | Merged the opt-in tiled CUDA scorer through `origin/main`. The scorer keeps FP32 scores and the same output hashes. |
| [jmnargi/Strata-V100 #105](https://github.com/jmnargi/Strata-V100/pull/105) | Merged opt-in CUDA host KV mirroring by DMA and verified it with the CUDA parity test. The source fork measured a component gain. Its full-model runs showed mixed results. |
| [chenjxdev/Strata-v100x2](https://github.com/chenjxdev/Strata-v100x2/commit/4a9a8ee5f) | The fork reverted its Q5_0 Split<6> and MTP-chain changes. Its V100 A/B found the first slower and the second within noise. |
| [GLCdz four-card results](https://github.com/GLCdz/Strata-V100/blob/dfdd99f0e/community/v100/BENCHMARKS.md) | Its long-context tuning uses the ISTA-DAS IQ3_S pack and 251.55 GiB of RAM. It changes several parameters at once. Those settings were left out of the 94 GiB Unsloth profile. |
| [StrataGP](https://github.com/gputier/StrataGP/commit/6d51272a2) and [wujungod/Strata-V100](https://github.com/wujungod/Strata-V100/commit/fb58e0dbc) | The transparent huge page patch and the older Volta stage-pinning fix are already in the merged history. The current allocator checks free huge pages to avoid compaction stalls. |
| [mavyric/Strata-v100](https://github.com/mavyric/Strata-v100/commit/3e046005d) and [RustingStrata-V100](https://github.com/gansui/RustingStrata-V100/commit/2beaf78b5) | The first patch addresses non-finite quantization values. The second enables Volta in a separate Rust backend. Neither supplies a portable measured speed gain for this engine. |

CPU A/B used one Xeon E5-2680 v4 core, GCC 11.4 at `-O3`, seeded token IDs, and the same inputs from both builds. There were three runs per arm in swapped order with seven samples per length in each run. The suffix test resets and appends the prompt into a table sized for 524,288 tokens plus 4,096 headroom. The prefix test mismatches one of the last 16 compared positions and checks identical returned prefix lengths.

| Tokens | Suffix reset and rebuild, ms before / after | Exact prefix match, µs before / after |
| ---: | ---: | ---: |
| 2,048 | 10.895 / 0.036 | 0.871 / 0.257 |
| 32,768 | 11.868 / 0.653 | 14.642 / 4.874 |
| 131,072 | 15.236 / 2.704 | 73.575 / 28.003 |
| 524,288 | 31.481 / 21.549 | 252.053 / 111.938 |

On one Tesla V100-SXM2-16GB with CUDA 12.9.86 and a 200 W limit, the merged scorer gave the following GPU-event times. Each median comes from three interleaved runs per arm, 20 iterations per run at 256 queries. Scores hashed identically at each context size. The switch stays opt-in. The benchmark deliberately selects the warp-scoring route.

| Context cells | Warp scorer, ms | Tiled scorer, ms | Time reduction |
| ---: | ---: | ---: | ---: |
| 32,768 | 0.861 | 0.680 | 21.0% |
| 131,072 | 3.469 | 2.524 | 27.2% |
| 200,000 | 5.420 | 3.667 | 32.3% |

Run `qsa_select_bench <context> 256 20` with `STRATA_QSA_WARP=1 STRATA_SELECT_SIMT=0 STRATA_SCORES_THREAD=0 QSA_BENCH_HASH=1`. Alternate `STRATA_SCORES_PROMPT_HALVE=0` and `1`. The figures above cover individual operations. They do not establish a change in model throughput.
