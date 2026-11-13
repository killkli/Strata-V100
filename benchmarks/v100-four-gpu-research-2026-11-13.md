# Four V100 deployment review and optimization research

Checked on 13 November 2026. The installed engine has useful Volta changes, but it does not include all current fork changes. The first candidates are the split HC_Q8 memory fix, stage-local host memory, and sufficient VRAM headroom. New drafting and tensor-parallel engines need separate workload and quality comparisons.

This review covers the local deployment, recent Strata discussions, other V100 projects, and NVIDIA guidance. It does not change the installed engine, model, server configuration, or operating-system settings. Machine-readable evidence and the transfer probe are in [the audit directory](../bench/results/2026-11-13-four-v100-audit/).

## Current deployment

| Item | Verified state |
| --- | --- |
| Installed engine | 0.1.45.7, built from fork commit `8b4c09bd` with CUDA 12.9 for `sm_70` |
| Worktree | `d189e410`, local deployment branch |
| Current fork head | `ee96b35e`, 12 November; 22 commits after the deployed source |
| Current upstream release | App/Terminal 0.2.13, engine 0.1.59; published 9 November |
| Hardware | Four V100-SXM2-16GB, each limited to 200 W; two Xeon E5-2680 v4 CPUs; AVX2 |
| Host memory | About 64 GiB on node 0 and 32 GiB on node 1 |
| GPU placement | GPUs 0/1 on node 0; GPUs 2/3 on node 1 |
| PCIe | Gen3 x16, x16, x8, x16 |
| GPU links | No NVLink shown. Peer reads report OK within 0/1 and 2/3, and TNS across sockets |
| Model | Earlier Unsloth UD-IQ4_XS, native pack; not the new v3.1 checkpoint |
| Expert formats | 43 layers IQ3_S/IQ4_NL, one IQ4_XS/Q8_0, four IQ3_S/Q8_0 |
| Serving configuration | Split `11,24,36`; two request slots; MTP spec 4, min-p 0.5; INT8 KV; 262144 context; 32768 resident KV cells; CPU vision |
| Explicit environment | `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0` |
| Process state | No text engine running at inspection |

Both deployed and current fork sources report **0.1.45.7**. Compare commit IDs, not only `--version`. The fork and upstream engine version series also differ. See [the fork head](https://github.com/jmnargi/Strata-V100/commit/ee96b35efdf7ddf3626f74c09c1d448be1479e77) and [the upstream release](https://github.com/Niko1221/Strata/releases/tag/v0.2.13).

The November 9 deployment records show ten passing API checks: health, model alias, authentication, OpenAI plain and streaming output, tools and tool results, Anthropic messages, Responses API, and two clients. A 66444-token needle and CPU image recognition also passed. These are historical checks, not a new live-server test or a filled-262K test.

That deployment already selected 15 Volta speed switches, stage-weight trimming, resident expert complements, and a reduced draft vocabulary. Its prompt plan used an 8192-token chunk and a 96-slot ring. It held about 82% of expert pairs in GPU caches at startup. Adding the same Volta switches again would not establish another gain. `STRATA_ADAPT_OVERLAP` was explicitly inactive on this layer split.

## Local transfer measurement

The probe used a separately mapped 128 MiB buffer. CPU 0 or 14 first touched every page. `cudaHostRegisterPortable` then pinned it. `numa_maps` confirmed all 32768 pages on the intended node. Each value is the median of three samples, with 12 asynchronous copies per sample and three warmup copies.

| Destination | Host pages on node 0 | Host pages on node 1 |
| --- | ---: | ---: |
| GPU0, node 0, x16 | 12.024 GB/s | 7.350 GB/s |
| GPU1, node 0, x16 | 12.013 GB/s | 7.361 GB/s |
| GPU2, node 1, x8 | 5.977 GB/s | 6.083 GB/s |
| GPU3, node 1, x16 | 8.726 GB/s | 12.193 GB/s |

This proves a bulk-transfer locality cost on this machine. It does not prove the running engine's page placement or an inference speedup. The November 9 startup probe measured 7.2, 10.9, 6.0 and 8.7 GB/s across the cards. Those older numbers are consistent with placement effects, but they do not identify the cause by themselves.

Stage-local allocation is therefore a strong candidate: place layers owned by GPUs 0/1 on node 0 and those owned by GPUs 2/3 on node 1 **before** page registration. Measure the real expert pages and CPU pool too. Keep the large PLE table and unequal node capacities in the budget. Binding the entire four-card process to one socket would make half the cards remote.

The current source has a NUMA free-memory guard, but no `STRATA_NUMA_POOL` implementation. Setting that newer upstream variable here would not add NUMA scheduling. Upstream's pool also has eligibility and source-path limits. See [upstream NUMA settings](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md) and [NVIDIA NUMA guidance](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#numa-best-practices).

IOMMU groups currently use DMA-FQ. However, local x16 copies already reach about 12 GB/s. The [single-V100 Swift recipe](https://github.com/koldfrontier/dgx-spark-ops/tree/main/recipes/Swift-1.5-IQ3_S-Strata-Single-Tesla-V100) measured a large `iommu=pt` benefit on another host. That is a lower-priority experiment here, not a proven fix. Check GPU2's slot and upstream link capability before treating its x8 link as a fault.

## Current fork changes worth testing

| Priority | Change | Local implication and evidence |
| --- | --- | --- |
| P0 correctness | [FP16 expert offset, PR 101](https://github.com/jmnargi/Strata-V100/pull/101), merged 12 November | Local `prefill.cpp` still reads `m.Hh` instead of the expert's offset rows. The new test detects the old bug. The report concerns UD-Q4_K_XL. Local IQ3_S/IQ4_NL and Q8_0 formats have MMQ support, so a failure of the current model is not established. Fix before testing FP16 fallback variants. |
| P1 memory | [Split HC_Q8, PR 102](https://github.com/jmnargi/Strata-V100/pull/102), merged 12 November | Local code excludes multi-GPU runs from canonical HC skipping. Its Q8 upload loop also lacks the new layer-range filter. Historical logs report 0.63 GiB of Q8 HC on every card. The patch should recover capacity; measure the four-card result. The author's two-card NVFP4 test recovered about 1.1 GiB versus its old Q8 path, not a guaranteed throughput gain. |
| P2 long context | [Exact compact top-k, PR 103](https://github.com/jmnargi/Strata-V100/pull/103), merged 12 November | `STRATA_TOPK_COMPACT=1` narrows exact QSA selection. The author's selector microbenchmark improves about 29–40%. Service results do not establish a stable end-to-end gain. Test at 64K and 128K, including short additions to long cached conversations. |
| P2 model choice | [QAD NVFP4 work](https://github.com/jmnargi/Strata-V100/blob/ee96b35efdf7ddf3626f74c09c1d448be1479e77/docs/NVFP4.md), 10–11 November | Adds checkpoint conversion and exact NVFP4 hot-path fixes. It changes the model and pack, so evaluate quality and memory separately from an engine-only comparison. |

The newer QPN2 NVFP4 kernel is not a universal win. The fork's own service comparison found its VRAM-only path slower than the default W4A8 path, despite a faster isolated benchmark. It remains off. This is direct evidence against enabling it solely because V100 has FP16 tensor cores.

## Recent discussions and other implementations

| Source | Useful direction | Applicability to this host |
| --- | --- | --- |
| [Three-stage V100, issue 5538](https://github.com/Niko1221/Strata/issues/5538), 7 November | On three PCIe V100s without NVLink, reach waits cost 8.4 of 30.3 ms per window. DFlash2 tuning and handoff costs matter after expert hits become high. | Close topology match. Overlap reduced waits but lost acceptance and 7% of decode speed; pipeline windows lost 10%. Profile accepted tokens/window as well as wait time. |
| [Layer-major prefill, issue 3976](https://github.com/Niko1221/Strata/issues/3976), comments through 12 November UTC | Fewer repeated expert reads can help SSD-bound prompts. | Read the late comments: engaged RAM-resident dual-card runs lost 29–44%; another forced run collapsed. Local planning reported almost no model-drive reads. Keep chunk-major as the control. |
| [Prompt-aware split, issue 5265](https://github.com/Niko1221/Strata/issues/5265) | Price expert bytes over each stage's link, not only estimated decode time. | Relevant to x8 GPU2 and unequal HC/head costs. Sweep nearby boundaries, preserve socket-contiguous order, and measure prefill and decode separately. |
| [Pipelined slots and MTP, issue 3794](https://github.com/Niko1221/Strata/issues/3794), updated 13 November | More slots, MTP and pipeline overlap can compete. | Local configuration has two slots. Compare solo latency and aggregate two-client throughput. The cited fully resident rig lost 26% with serial batch MTP; it is not a local prediction. |
| [IQ3_S grouped GEMM, issue 5569](https://github.com/Niko1221/Strata/issues/5569) | SIMD dequantization and a smaller-row tile cut measured kernel time by 24–36%. | Local model uses IQ3_S heavily. Confirm the dispatched kernel first. The author's service speed was neutral, and the patch does not optimize every expert path. |
| [Volta attention storage, issue 5523](https://github.com/Niko1221/Strata/issues/5523), 7 November | Shared-memory reuse raises FP16/K8V4 attention occupancy. | Local KV is INT8. INT8 gained only 2.3% in that kernel; forcing three blocks spilled registers and slowed it. Do not change KV format to chase this kernel result. |
| [1Cat-vLLM 1.5.2](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.2), 6 November | V100-specific TP, QSA selection, host KV, concurrent speculation and prefix-cache work. | Release tests use four **32 GB** cards with NVLink and TP4. Some headline decode numbers assume 70% acceptance. This 64 GB, cross-socket PCIe host needs a new benchmark. The compact-selector idea has already reached the fork. |
| [v100-skinny](https://github.com/dnv2003/v100-skinny) | QPN2/QPN8 weight layout and verify-width tuning on four 16 GB V100s. | Its main model is dense Qwen3.8-27B, not this mixed-format Flash-Next MoE pack. Use it as a kernel/design reference or a smaller-model service candidate. |
| [Strata AC922](https://github.com/eelgaev/Strata-AC922/blob/ac922/docs/IBM_AC922.md) | Per-node pinned arenas, helper affinity, chunk pipelines and peer fetching. | POWER9 has coherent CPU–GPU NVLink. Borrow the placement and scheduling ideas, not its bandwidth thresholds or speed targets. It is older than the November work. |
| [llama.cpp-v100 2.0.1.2](https://github.com/1115714829/llama.cpp-v100/blob/master/README.en.md), 3 November | Alternative V100 engine, topology-aware execution and same-socket model tests. | Much evidence is AC922/NVLink. Flash-Next single-slot tests do not establish this deployment's two-client API, tools and cache behavior. |
| [ExTV's asymmetric Strata](https://github.com/ExTV/strata-5090-4070/blob/master/README.md), 10 November | Separate prompt placement from cache capacity; reduce disk reads and improve agent reuse. | Its 5090 + 4070 and 31 GB RAM are different. Local data already fit RAM/VRAM. Investigate prompt placement only after measuring streamed bytes. |
| [Strata HotRod](https://github.com/sangharshadhyeta/Strata-HotRod) | Learned SSD lookahead and tiered experts. | Lower priority while local disk reads are low. Expert substitution changes computation and needs independent task-quality checks. |

New applications include newer Unsloth v3.1 sets, QAD NVFP4 checkpoints, and DFlash2 on newer upstream engines. The current fork already includes v3.1 format support but the configured model files are the earlier revision. Model replacement is a separate experiment. The local source has no DFlash2 implementation. Test newer DFlash2 engines in isolation, with their actual slot, drafter, graph-memory and pipeline constraints. A flag alone cannot add it to this engine.

## Recommended experiment order

1. Establish the unchanged four-card baseline. Preserve INT8 KV, context, model files, two-client functionality and CPU vision. Save an expert-profile copy and isolated conversation caches for each arm.
2. Test the FP16 correctness and HC_Q8 memory patches. Record native/canonical bytes, cache slots and free VRAM on all four cards. Run the new FP16 regression, tools, images, cancellation and concurrent stream-isolation checks.
3. Test stage-local first-touch and registration. Verify page placement after pinning. Compare host-node bandwidth, CPU waits and GPU reach waits. The historical default used 13 workers on socket 0; increasing worker count without source locality can lose speed.
4. Resolve headroom and PLE locking. Historical free VRAM was 422/516/514/516 MiB before further graph capture. The PLE `mlock` failed. Current shell memlock is about 11.8 GiB. Price PLE, expert pinning, KV and graph growth separately; do not assume CUDA page registration and `mlock` have identical limit accounting. Start reserve experiments near the logged suggestion of 832 MiB and inspect every stage. Verify the actual PLE table size before changing service limits.
5. Compare the current split with `12,24,36`, then small boundary changes around GPU2 and the last-stage head. Keep `0,1,2,3` as the topology control. Test 4096 versus 8192 prompt chunks and 96 versus 192 ring slots independently. These are search candidates, not recommended new defaults.
6. Separate interactive and concurrent goals. Compare one versus two slots, then spec depths 2/4 and min-p 0.5/0.7 with the compatible scheduling mode. A one-slot experiment is a latency alternative; it does not replace the required two-client service. Test compact top-k separately on long contexts.
7. Consider per-NUMA GPU-pair replicas only if the selected model fits their VRAM/RAM budgets. This could improve independent-request locality, but may reduce cache residency and duplicate model/KV memory. It requires a routing layer and a full comparison with the four-stage baseline.

Use interleaved A/B/B/A arms with a fresh server per arm and at least three repeats. Record actual prompt tokens, reused tokens, first visible output latency, completion latency, decode rate, accepted/offered drafts, total concurrent throughput, page placement, CPU waits and VRAM minima. Include 2K/8K/32K cold prompts, 64K/128K context probes, agent continuations, code/JSON, Chinese prose, tools and images. Do not count prefix reuse as faster cold prefill. Report changed answers and quality checks with the speed result. Use [the existing prefill runner](../bench/run_v100_prefill.py) and [the agent-turn runner](../tools/agent_turns_bench.py) where their scope matches.

Do not copy the fork's dual-data-parallel prefill experiment into this profile without adaptation. Its tested port requires two GPUs, no batch or stage trim, and host-streamed KV. This deployment has four stages, two slots and trimmed weights. The fork's separate daily NVFP4 test used elastic GPU KV and declined that port for a different reason.

## Verification and reproduction

Four existing CPU tests passed again on November 13. GPU0's existing `qsa_prompt_attn_parity 32768 2048 5` completed with zero failures. Its INT8 32K attention error versus FP64 was `2.33e-6`; the old/new synthetic timings were 24.079/12.414 ms. Q4_0 cases were skipped below sm_80. This tests the current attention implementation, not the new compact-selector patch or the whole model.

To reproduce the host-transfer test, run from the repository root with CUDA 12.x on `PATH`. CPU IDs 0 and 14 in this probe belong to the two local sockets; adapt them on another machine.

```sh
nvcc -O2 -arch=sm_70 \
  bench/results/2026-11-13-four-v100-audit/numa_h2d.cu \
  -o /tmp/strata-numa-h2d
/tmp/strata-numa-h2d
```

No new full-model speed claim follows from these checks. Keep CUDA 12.x and `sm_70`; NVIDIA identifies CUDA 12.9 and driver R580 as the toolchain/driver path for Volta. See [NVIDIA's architecture-support guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers).
