# Four V100 upstream integration and NUMA performance

The selected four-GPU profile improves fresh-prompt processing on the earlier Unsloth UD-IQ4_XS Flash-Next model. It combines the current upstream engine with the fork's Volta kernels, stage-scoped dense weights, and explicit host-memory placement. The model, INT8 KV, 262144 context limit, and two-client configuration stay the same.

The [external research review](v100-external-optimization-research-2026-11-13.md) covers independent V100 engines, current papers, and the next experiments. The [initial audit](v100-four-gpu-research-2026-11-13.md) records the deployment before this integration. Raw measurements are in [the result directory](../bench/results/2026-11-13-four-v100-upstream/).

## Hardware and source revisions

| Item | Value |
| --- | --- |
| GPUs | Four Tesla V100-SXM2-16GB, SM70, 200 W each |
| Links | PCIe Gen3 x16, x16, x8, x16; no NVLink |
| Placement | GPUs 0/1 on NUMA node 0; GPUs 2/3 on node 1 |
| CPU | Two Xeon E5-2680 v4; 28 physical cores; AVX2 |
| RAM | About 64 GiB on node 0 and 32 GiB on node 1 |
| CUDA and driver | CUDA 12.9; driver 580.178.04 |
| Original deployed engine | 0.1.45.7, source `8b4c09bd3d9213b2d47880826c6e818c3140664d` |
| Starting worktree | `d189e4103619341863f2fba77b28f8a71d32cb61` |
| Merged V100 fork | `ee96b35efdf7ddf3626f74c09c1d448be1479e77` |
| Merged official upstream | `13eedf552417a6a791a7d13c6597e24300ee3a09`, engine 0.1.59.1 |
| Final installed build source | `f271dbaeb`; source fingerprint `cc16d1a9befc4cfc`; final engine change `22a07ca38` |
| Model | Earlier Unsloth UD-IQ4_XS Qwen3.8-Flash-Next; unchanged weights and native pack |
| Service | Four stages, split `11,24,36`; two clients; MTP depth 4, min-p 0.5; INT8 KV; 32768 resident KV cells; CPU vision |

This is a CUDA SM70 and CPU-vision validation. HIP, SYCL, Metal, Windows, and mixed GPU architectures have no runtime result here.

Development uses separate commits for the merges (`707dd16f5`, `57358e427`), stage page placement (`c3a8610a9`), host preference (`1c387fe0e`), TurboQuant prompt and verify wiring (`e36c42db0`, `22a07ca38`), shared-state admission (`6bb2f9b33`), native disk caching (`13b3d80ab`), corrupt-prefix removal (`fad2f93fb`), and heap reclamation (`506a3b825`). Build and source fingerprints are in the result manifest.

## Warmed fresh-prompt comparison

Each size has three deterministic prompt variants. The engine generates 256 tokens per request. Every measured request has zero prompt reuse. Six earlier workload requests warm the adaptive expert cache. The timed variants use new case numbers 3–5. The harness records output text, answer hashes, actual token counts, draft counts, and HTTP wall time.

| Profile | 2105-token prompt, tok/s | 8576-token prompt, tok/s | 2105-token wall time, s | 8576-token wall time, s |
| --- | ---: | ---: | ---: | ---: |
| Original deployed engine | 715.8 | 1110.4 | 5.394 | 10.206 |
| New engine, host node 0 preference, CPU-row expert placement | 798.4 | 1153.2 | 4.993 | 9.793 |
| New engine, host node 0 preference, GPU-stage expert placement | 843.7 | 1202.9 | 4.870 | 9.486 |
| Installed selected profile, before compatibility follow-up | 844.0 | 1207.3 | 4.740 | 9.466 |
| Final installed profile, after compatibility follow-up | 799.1 | 1177.7 | 5.030 | 9.668 |

The final installed run improves median prompt rate by **11.6%** and **6.1%** against the original engine. Median wall time falls by **6.7%** and **5.3%**. These are the final deployment measurements.

The three intermediate profiles use source `1c387fe0e`. In that comparison, GPU-stage placement improves prompt rate by 5.7% and 4.3% against CPU-row placement. The earlier installed repeat is faster than the final run. The later run includes compatibility changes and took place in another session. This experiment does not isolate the cause of that difference. All runs remain in [the paired summary](../bench/results/2026-11-13-four-v100-upstream/summary.json).

The original median decode rates are 107.4 and 106.5 tok/s. The final installed run gives 107.6 and 108.0 tok/s. These small samples and differing answers do not establish a general decode gain.

The workload converts synthetic CSV rows into Python. None of the six warmed answers are byte-identical across engine revisions. This is a fixed-output-length HTTP comparison. It is not a token-identical kernel benchmark or a task-quality study. The arms ran in separate sessions, so clock, placement, and cache history can affect small differences. The results do not supply a confidence interval.

Early original-engine requests had much lower decode rates while the adaptive cache warmed. Those requests remain in the raw data. They do not support a twofold or threefold steady-state gain. The early disk-enabled baseline pilot is excluded from these tables.

## Memory placement and capacity

The upstream HC_Q8 path and the restored stage-scoped loading avoid repeated dense uploads on each card. The original GPU expert caches have 20167 slots in total. The selected four-stage service has 21293 slots, an increase of 1126 or 5.6%. Its host expert complement falls from 24.94 GiB to 22.40 GiB. Both profiles use a 96-slot prompt ring.

The selected environment is opt-in:

```json
{
  "STRATA_HOST_NUMA_NODE": "0",
  "STRATA_NUMA_STAGE_LOCAL": "1"
}
```

The first setting gives future host allocations a preference for the roomier node 0. It leaves CPU affinity unchanged. Expert helper threads then restore local memory policy and touch each stage's pages near its GPU. The page plan includes only whole, disjoint pages. The caller handles boundary pages.

The selected startup places about 10.41 GiB of expert pages on node 0 and 11.96 GiB on node 1. The final installed startup again reports 100% on the intended node for both groups. Its free VRAM after loading is 684, 778, 778, and 558 MiB. An earlier process page-map sample places almost all of the 24.34 GiB resident PLE mapping on node 0. The expert complement is split across the two nodes. The [final startup extract](../bench/results/2026-11-13-four-v100-upstream/installed-final-startup-placement.txt) preserves the observed placement and capacity.

The policy applies to new allocations. Existing file-cache pages retain their physical placement. For the controlled host-preference comparisons, all temporary model processes stopped before a file-specific `POSIX_FADV_DONTNEED` hint on the selected model shards. This did not change file bytes. It did not change global kernel cache settings. [Linux NUMA policy](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html), [set_mempolicy](https://man7.org/linux/man-pages/man2/set_mempolicy.2.html).

Stage-local placement alone failed in an earlier arm. Node 1 had about 6.28 GiB free for an 11.96 GiB allocation. Actual locality reached only 61.2%. Decode became slower. That arm also changed GPU0's startup transfer probe and automatic PCIe fraction, so it cannot isolate one NUMA effect. Its data remain in the result directory. Do not deploy that profile in place of the two-setting profile above.

The memory-policy test skips when either node has less than 512 MiB free. Local policy can spill under real pressure. The no-pressure test passed. The startup placement report and available memory remain useful checks on later runs.

## Merge compatibility and correctness

The integration retains the fork's Volta FP16 activation path, grouped WMMA experts, format decoders, architecture defaults, per-device elastic KV ownership, and shared conversation payloads. It uses the newer official request-slot and snapshot restore guards.

The final compatibility work reconnects resident TurboQuant choices, both prompt writers, and serial and per-slot verification writers. KV streaming remains unsupported with TQ. It also reconnects the opt-in native disk conversation tier and its append-only records. Disk caching requires an explicit path and budget. DFlash2 state is not supported by that experimental tier. The saved service profile uses INT8 KV and no disk tier.

A separate real-model TQ6 profile passed all 11 API checks and three rounds of two-client streaming responses, cache continuation, and historical-branch isolation. It used four GPU stages, two request slots, a 16384-token context limit, and disabled KV streaming (`--kv-resident 0`). This is a compatibility result. Its context and KV residency differ from the INT8 service, so it supplies no TQ speed comparison.

Three real-model native disk-cache scenarios passed. An A/B/A+ request sequence reused 249 tokens with output and recurrent-state parity against the cold control. A process restart recovered the records and preserved reuse, output, and state. A damaged first panel failed its checksum, removed the corrupt chain and dependent records, and produced the cold-control answer without reuse. Disk caching stays off in the saved service profile.

Shared-checkpoint admission now counts physical allocations once. Snapshot capture honors an explicitly supplied historical checkpoint instead of reading newer recurrent state from the live session. Disk prefix reads discard corrupt chains and their dependent branches before a later request can select them again.

Eviction also retains the newer upstream heap trim. In the RAM-return test, eviction first returned none of a 512 MiB entry to the operating system. With the fix, RSS fell by about 511 MiB. The next physical-memory admission check can observe those released pages.

## Validation scope

The final installed profile passed all 11 API checks. It passed a 66444-token needle, a CPU-encoded red-image query, six initial workload requests, six fresh warmed requests, and three rounds of two-client streaming, continuation, and historical-branch isolation. The cache checks observed reuse. Both clients kept distinct output. The [validation record](../bench/results/2026-11-13-four-v100-upstream/installed-final-validation.json) links each phase's results.

The context setting remains 262144. The longest completed retrieval probe here has 66444 prompt tokens. It does not validate a filled 262K conversation.

The final cache-focused CPU set passed nine tests. The GPU set passed nine tests, including grouped-GEMM parity against FP64, BF16 GEMM, 112 production W16 cache combinations, TurboQuant kernel parity, device-local VMM properties, and host-memory policy. The CPU set includes conversation-cache checks, 1925 disk-store checks, NUMA page planning, file storage, transfer validation, and heap reclamation.

After the TQ verify-writer fix, five TQ and snapshot tests passed. A further regression check captures an older checkpoint while the live session has newer state. It passed in synchronous, asynchronous, and VRAM-checkpoint modes. The shared CUDA kernel library was unchanged by the verify-writer fix. The disk gate's 20 offline tests also pass. Raw test summaries are in the result directory.

Server tests ran 135 cases, with one skipped. Terminal testing covered large portions of the suite. One segment reached 1704 passes and 24 skips before interruption. It also found a discovery-fixture failure; the corrected targeted set passed 22 cases. A second segment reached 1413 passes and 30 skips with four stale-fixture or packaging failures. Targeted fixes passed afterwards. Exhaustive UI walks did not complete. This is not a claim that the complete Terminal suite passed.

## Reproduction and rollback

Use private copies of the service configuration and expert profile for each arm. Keep the same model, GPU order, split, context, and generation length. Run the API and long-context checks before timing. Run initial and warmed workloads separately. Stop other compilers and test runners before timing.

The tools are [the HTTP verifier](../tools/verify_v100_api.py), [the long-context and image verifier](../tools/verify_v100_long_and_vision.py), [the streaming cache verifier](../tools/verify_v100_cache_isolation.py), and [the result analyzer](../bench/analyze_v100_upgrade.py). Use `--repeat-start 0 --repeats 3` for the first workload pass and `--repeat-start 3 --repeats 3` for fresh warmed cases. The result analyzer rejects changed prompt counts, reused prompt tokens, incomplete outputs, and unpaired case numbers.

The [deployment patch](../deploy/v100-x99-upstream-20261113.patch.json) adds only the two selected environment values after checking the expected profile. The [rollback helper](../rollback-strata-upgrade.sh) verifies the saved component checksums before replacing files. It preserves authentication, network settings, context, and unrelated user settings. `--check` validates the backup without changes. `--no-start` restores it while leaving the service stopped.

An install, rollback, and reinstall round trip restored all four original components and the original configuration, then reapplied the selected profile. The final installed component hashes match the packaged build. Temporary validation servers used localhost and stopped after the checks. The service was idle at inspection and remains stopped. Authentication, network settings, model paths, two slots, and the 262144 context limit are preserved. The [completion audit](../bench/results/2026-11-13-four-v100-upstream/completion-audit.json) records the final hash, configuration, and artifact checks.
