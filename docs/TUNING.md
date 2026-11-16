# Performance and tuning

This page shows how to make Strata faster on one PC. It gives the method to measure a
change, the settings that matter for each goal, the optimization catalog of the engine,
and the settings that were measured and rejected.

Each number comes from this repository, and each claim names its hardware. Do not move
a number to other hardware. First install and choose the model that fits the PC:
[INSTALL.md](INSTALL.md), [MODELS.md](MODELS.md), [QUANTS.md](QUANTS.md). Terms:
[GLOSSARY.md](GLOSSARY.md). Flags and defaults:
[ENGINE_FLAGS.md](ENGINE_FLAGS.md). Environment variables: [ENVIRONMENT.md](ENVIRONMENT.md).
All measured runs: [DETAILS.md](DETAILS.md). How the engine works:
[HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## How to measure

### Fix one workload

Change one setting at a time. Keep every other input equal.

| Hold constant | Why |
|---|---|
| the prompt set and its token counts | decode speed follows the text, through the draft acceptance |
| greedy sampling (`temperature 0`) | a sampled answer changes which drafts are accepted |
| the output cap (256 tokens is the common cap in this repository) | a short answer cannot separate two arms |
| a cold prompt (`reused=0`) | a reused prefix reads fewer tokens |
| the expert-cache slot count | another count places experts on the GPU differently |
| a fresh engine for each arm, for a restart-only setting | the engine reads those switches at start |

### Warm up, then interleave the arms

Discard the first request. The first read of a prompt is cold, and the calibration in
`strata_control/tune_session.py` discards one warm-up and then measures three decodes of
200 tokens. The RTX 3090 Ti prefill table also leaves out the cold first pass
([MULTI_GPU.md](MULTI_GPU.md)).

Run the arms in a fixed order, and alternate the sides: A-B-A, A-B-B-A or A/B/A/B/A/B. For
a restart-only setting, start a new engine before each measured arm. Report the pairs that
the candidate won, not only the median.

Most switch tables in this repository use five to eight interleaved pairs. The
per-architecture tables used eight pairs, and the Volta switch studies used six.

Prompt speed repeats well when the text, the chunk size and the cache state do not change. A
32K prompt took 14,051, 14,071 and 14,048 ms in three arms of one study (two Tesla
V100-PCIE-16GB). Keep the same prompt text, report its hash, and disable prompt reuse.

### Watch one metric for each claim

| Metric | What it answers | Where to read it |
|---|---|---|
| prefill tokens/s | how fast the engine reads a fresh prompt | engine log and `/metrics`: `prompt_read`, `prompt_ms`; `STRATA_PREFILL_TIMING=1` for the parts |
| decode tokens/s | how fast the engine writes tokens | the engine's decode timing, and `drafts_offered` / `drafts_accepted` |
| time to first token (TTFT) | how long the user waits for the first token | `/metrics` `ttft_ms`; the server's streaming value includes parser buffering |
| total latency | the whole request | wall time; this value is not a decode rate |

Do not divide generated tokens by the total request time. That value contains the prompt
read and the queue wait.

### A decode claim under 5% needs several samples

A change below 5% is not a result from one pair. Require two conditions: every pair or most
pairs agree, and the gap is larger than the drift of the control arm.

Two Tesla V100-PCIE-16GB cards at 175 W unless the row says otherwise.

| Evidence | Number | Hardware |
|---|---|---|
| one arm's own repeats, 32K prompts, 256 output tokens | 59 to 94 tokens/s | two Tesla V100-PCIE-16GB ([DETAILS.md](DETAILS.md)) |
| two control arms of the same setting, drift over one run | 8% apart | two Tesla V100-PCIE-16GB |
| run-to-run floor of the teacher-forced KL, same engine twice | 0.0006-0.0007 nats | RTX 3060 12 GB, RTX A4000 16 GB |
| a 5.2% decode difference, called indicative and not a guarantee | 120.2 to 126.4 tokens/s | two Tesla V100-PCIE-16GB ([NVIDIA_V100.md](NVIDIA_V100.md)) |

The tuner applies the same rule: a candidate wins only with 3% or more in every round
(`strata_control/tune_knobs.py`).

### Instruments

| Instrument | Output | Cost |
|---|---|---|
| `GET /metrics` | `ttft_ms`, drafts offered and accepted, `hit_rate`, `pcie_share`, `ram_blobs`, `file_blobs`, `file_mb` | none |
| `--stats` | the decode's tiers: RAM blobs, file blobs, MB read, time in the reads | none |
| `STRATA_PREFILL_TIMING=1` | where each prompt chunk's time goes | none |
| `STRATA_DECODE_TIMING=1` | the per-window split: GPU wait, CPU pool, stage, commit, draft | small |
| `STRATA_VERIFY_PROFILE=1` | the GPU's own stamps for each stage | slows decode; use it to compare, not to measure speed |
| `--window-hashes` | one line per verify window, for a same-arithmetic check | none |
| `--spec-follow PATH` | the same continuation in both arms, so both process the same text | use with `--spec` |
| `--adapt-every 0 --prompt-cache 0 --pcie-frac 0` | byte-identical repeats of the same request | slower; A/B only |

Record the hardware, the engine version, the model and quantization, the context, the KV
format, and the cache state for each arm. The full list is in
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) and [benchmarks/README.md](../benchmarks/README.md).

## The settings that matter, by goal

Every row gives the default from the code (`src/program/generate.cpp`) or the setup value.
A setting that changes the answers is marked.

### Decode speed

Hardware is named in the measured cell of each row.

| Setting | Default | Effect | Measured |
|---|---|---|---|
| `--mtp` with `--spec N` | `--mtp` off, `--spec 0`; setup writes `--spec 4` | the draft layer proposes N tokens per verify window; an accepted draft costs no full pass | RTX PRO 5000 Blackwell (110 SMs), 3 pairs: with the cheaper rows, `--spec 4` measured 1.122 / 1.141 / 1.132 of the earlier engine (story / code / document), `--spec 8` 1.027 / 1.105 / 1.069 |
| `--spec-min-p P` | `0.0` in the engine; setup writes `0.5` | the draft layer must be this sure to extend the window; trims weak drafts per token | Arc A770, 546-token prompt, 256 tokens: `0.0` 10.4 tokens/s, `0.5` 12.2, `0.7` 11.8 |
| `STRATA_ROUTE_TAIL_SKIP` | `7` (on) on CUDA, one GPU or a split; off with batch slots, a peer tier, a helper GPU, or 90% or more of the experts cached | a missed expert that every window token ranks 7 or lower is not copied or computed; the other weights stay as the router gave them; the answers change a little | default over `0`: 1.156x RTX 3060 12 GB, 1.129x RTX A4000 16 GB, 1.146x Tesla P100 16 GB, 5/5 pairs each; 0.998x on two RTX A4000 with a 99.7% hit rate |
| `--expert-cache N` (0 = auto) and the adaptive tier `--adapt-every`, `--adapt-swaps`, `--adapt-decay`, `--adapt-min-gain` | auto; `4`, `96`, `0.7`, `1.5` | more slots keep more experts on the GPU; the tier swaps experts in and out | Tesla P100, IQ2_XS, 64K prompt: 2,083 slots 27.5 tokens/s, 3,500 30.9, 5,000 34.2, 7,000 36.7 |
| `STRATA_ADAPT_OVERLAP` | opt-in; on by default on one Volta card on Linux CUDA; never on a layer split | the tier's copies land while the next window runs | RTX A4000, 6 pairs: Q2_0 16 GB 62.5 to 66.2 tokens/s (+5.9%), IQ3_XXS 16 GB 48.6 to 52.6 (+8.3%); RTX 5090 Laptop loses 21% (84.6 against 106.6), which is why it is opt-in there |
| `--kv-resident N` | `0` (all K/V in VRAM) | keeps N cells of each attention layer in VRAM and the rest in pinned RAM; the freed VRAM goes to expert slots | R9700 at 128K: +5.4%, 5/5 pairs; Tesla P100 at 64K: 34.05 / 34.06 / 34.21 / 34.92 tokens/s at 4,096 / 16,384 / 32,768 / 65,536 cells |
| `STRATA_VERIFY_ECAP=N` | off; changes the answers | caps the expert set of a verify window: token 0 keeps all its picks, the tail keeps the best N-K by weight | two V100, 32K, 512 greedy tokens: +17.7% at `N=20`, +16.2% at `N=16`; the prompt path did not move |
| `--draft-vocab SUB` | the subset with every CJK token | the drafter's token subset; a smaller head takes less VRAM | CJK answers 15-38% faster (Q2_0, RTX 5070); `en` 1-2% faster English and about 110 MiB less VRAM; `fr` 141 to 158 tokens/s in French (RTX 5090) |
| `STRATA_MMVQ_IL` (table per card) | on for sm_80 and newer; the sm_70 table turns it on for Volta; off on sm_75 and older | the dense 2-8 token projections read an interleaved copy of the activation | RTX PRO 5000 Blackwell: the Q5_K head at 5 tokens 899 to 491 us; the 5-8 token table is sm_120 only |

### Prompt speed

Hardware is named in the measured cell of each row.

| Setting | Default | Effect | Measured |
|---|---|---|---|
| `--prefill CHUNK` | auto, up to 8192 | a larger chunk streams each expert once for more tokens, and borrows more of the expert cache for its buffers | two V100, split 16, IQ3_S: 16384 read a 32K prompt 7.5% slower (1,607 against 1,736 tokens/s), 4,096 23% slower, 2,048 54% slower; 12-16 GB cards at 20K: 16384 -3.4% to +4.5%, 32768 +4.5% to +7.7% |
| `STRATA_PREFILL_HOST_PLAN` | on, Linux, file tier (`=0` turns it off) | picks the chunk with the fewest predicted reads of the expert files | RTX 3060, IQ2_XS, 450 MB/s emulated SATA: 334.0 to 822.3 tokens/s at 20,400 tokens (2.46x), 5/5 pairs |
| `STRATA_PREFILL_CPU_SHARE` | `auto` below 1,024 tokens, CUDA, one GPU | the CPU pool computes the experts that few of the chunk's tokens route to | 512 tokens: -26% to -35% prompt time (RTX 5070, RTX 3060, Tesla P100), 8/8 pairs |
| `--prefill-head-alias` | off | the output head's VRAM holds the prompt buffers during the read | 8 GB RTX PRO 2000 Laptop: 292.5 to 667.1 tokens/s on a 29,854-token prompt ([PREFILL_HEAD_ALIAS.md](PREFILL_HEAD_ALIAS.md)) |
| PCIe share: `--pcie-frac`, `STRATA_PCIE_FRAC_DEFAULT` | auto since 0.1.42, from the CPU pool's measured speed | the share of missed experts that the GPU reads over PCIe; the rest runs on the CPU | RTX 3060: 0.55 to 0.20 gave 1.300x decode, 5/5 pairs; Tesla P100 1.011x |
| `--pcie-mode dma` | `auto` (the copy kernel) | copies the PCIe share on a second stream, with a device-raised flag | two V100: 15.39 / 15.79 ms per token with the copy kernel, 14.58 / 14.53 with DMA (-6.6%), +8.1% tokens/s |
| `STRATA_SPLIT_MTP_BATCH` | off | batches the draft layer's prompt K/V on a layer split | two V100: 8K 5,524.5 to 5,265.4 ms, 32K 15,383.4 to 14,348.5 ms |
| `STRATA_SELECT_SIMT` | off in the engine; on in the V100 profile | tiled FP32 prompt scorer on pre-sm_80 CUDA cards | two V100: 65K 27.609 to 26.212 s (+5.3%), 131K 58.216 to 51.909 s (+12.2%) |
| `--prefill-pipe R` | `0` (off) | chunk count sized for a split's pipeline, from the rig's b/a in tokens | four GP100 dies: +62% (355 against 219 tokens/s); two RTX 3090 Ti: keep it off, b/a 768 was 20% slower |
| `STRATA_PREFILL_PROMOTE=N` | `0`; the V100 profile uses `2048` | hot experts stay in stable slots after a prefill loan | two V100, nine agentic turns: TTFT median 1,176.9 to 987.3 ms; the smallest turn's full request was 7.6% slower |
| `STRATA_PREFILL_STREAM_MIN` | `1024`; `2048` on the eligible V100 tensor-core plan | the read length below which the prompt path stages experts, and can hand them to the CPU pool | two V100 with the cutoff raised from 3,072 to 6,144: a 3,072-token prompt 2.639 to 2.265 s (+16.5%), 4,096 3.558 to 2.876 s (+23.7%) |
| `STRATA_SM70_CHUNK` | the sm_70 rule: a bare `auto` can take 16384 on one Volta card with a native pack whole in RAM; `0` keeps 8192, `1` forces 16384 | a larger automatic chunk on one Volta card | no isolated number; the rule keeps 8192 when the RAM does not cover the larger loan plus 6 GiB, and a layer split keeps 8192 |

### Low-RAM PCs

Hardware is named in the measured cell of each row.

| Setting | Default | Effect | Measured |
|---|---|---|---|
| `--resident-experts` | off (setup selects it) | page-locked RAM copy of the experts that the GPU cache lacks; no SSD read while the engine answers | `STRATA_EXCHANGE_ROTATE=1` on the same mode: 163.4 to 187.4 tokens/s (RTX 5090, Swift 1.5 IQ3_S, 8K prompt) |
| `--mmap-experts` | off (the resident arena) | maps `experts.bin`; the OS file cache holds the experts | with `--mmap-experts` and the RAM capped at 24 GB, `STRATA_STAGE_PIN=1` gave 13.9 to 15.0 tokens/s (Tesla P100); the same read was 71.97 against 34.78 ms a token cold against warm (the source does not name the card); the pin stays opt-in because a 0.1.41 gate found IQ3_S decode corruption |
| `--kv int8` | `--kv fp16` | half the KV VRAM of fp16 (`int8` 1,056 bytes a cell) | memory only; this repository has no speed number for fp16 against int8 |
| `--kv-resident N` | `0` | the whole K/V lives in pinned RAM, about 13.7 KB a context token | see decode speed: R9700 +5.4% at 128K, Tesla P100 flat at 64K |
| `--conversation-cache-mib N`, `--conversation-cache-slots N`, `--conversation-cache-min-free-mib N` | `0` (off), `4`, `2560` | keeps parked conversations in RAM | Tesla P100, 262K document: cold read 904 s, then 0.53 s median for questions 2-20; +804 MiB of checkpoints ([RESEARCH_RUNS.md](RESEARCH_RUNS.md)) |
| `--prompt-cache N`, `--prompt-cache-every N` | `6`, `16384` | checkpoints of the conversation, about 118 MB each | a server request without a turn boundary at the document's end re-read up to 16,384 tokens: 55.5 s each, against 0.53 s (Tesla P100) |
| `--prompt-cache-tail` | off | one extra checkpoint at a chunk boundary near the prompt's end | no number in this repository; the gain depends on where a branch leaves the prompt ([PROMPT_CACHE_TAIL.md](PROMPT_CACHE_TAIL.md)) |
| `STRATA_CACHE_MESSAGE_BOUNDARY` | off | checkpoints the previous turn's boundary when the last user message changes | two RTX 3080 20 GB: first complete edit 17.236 to 11.527 s (-33.1%), with 16K against 32K chunks |
| `--vram-elastic` with `POST /v1/vram` | off | gives expert-cache segments back to another program and maps them again | no speed claim; the call reports the measured free VRAM ([VRAM_ELASTIC.md](VRAM_ELASTIC.md)) |

### Long context

Hardware is named in the measured cell of each row.

| Setting | Default | Effect | Measured |
|---|---|---|---|
| `--max-context N` | `4096` | the K/V capacity; the cost is the bytes a cell takes | calculated, not measured: `int8` takes 1,056 bytes a cell, so 524,288 tokens on 12 layers is 6.1875 GiB |
| `--rope-scaling linear` or `yarn` | the model file's rope keys, else `none` | extends the context past the trained window; a quality setting | no speed number; fixed at start |
| `--kv-grow` | off | the K/V takes VRAM only for the cells the requests reach; the expert cache takes the rest; needs VMM | two V100: the caches grew from 6,199 to 7,333 slots (card 0) and 2,836 to 4,843 (card 1); the combined profile read 32K prompts +20.7%, a gain not isolated to this setting |
| `--kv k8v4` | off | 8-bit keys and rotated 4-bit values, 816 bytes a cell | RTX 3090, 198K context: 99 instead of 85 tokens/s, prompts 2-5% slower |
| `--kv q4_0` | off | 4-bit K/V after a Hadamard rotation, half of int8's memory; a quality setting | Q2_0 at 128K with `--kv-resident 32768`: 62.4 to 67.4 tokens/s and 1.52 to 0.83 GiB of K/V in RAM (the source does not name the card); perplexity +8% at 1K and +12% at 8K |
| `--kv tq3`, `tq4`, `tq6`, `tq8` | off | TurboQuant K/V, 392 to 1,032 bytes a cell; no `--kv-resident` | two V100, 51,746-token prompt: `int8` 24.289 s, `tq4` 35.190 s, `tq6` 39.272 s, `tq8` 41.085 s; the opt-in `STRATA_PROMPT_ATTN_TQ=1` reader cut `tq6` to 31.721 s and stayed 22.8-23.3% slower than `int8` |

### Several requests

Hardware is named in the measured cell of each row.

| Setting | Default | Effect | Measured |
|---|---|---|---|
| `--batch N` (`--slots N`) | `0` (off) | N conversations decode in one window, so the dense weights and the shared experts are read once for all of them | a slot takes 0.56 GiB at 32K with 8-bit KV; alone on a 12 GB card a request runs 11-24% slower; 8 clients on 2 GPUs: 109 against 78 tokens/s |
| `--batch-mtp` | off | each slot also verifies one MTP proposal per window | two RTX 3080 20 GB: 89.2 against 77.3 tokens/s aggregate; two R9700 with every expert in VRAM: -21.9% |
| `--batch-groups G` | `auto` on a split: one group per stage | the slots flow through the GPUs as a pipeline | 8 clients: 4x R9700 166 against 86 tokens/s; 2 GPUs 109 against 78 |

Details, the slot cost and the admission rules: [BATCHING.md](BATCHING.md).

### Two GPUs

Hardware is named in the measured cell of each row.

| Setting | Default | Effect | Measured |
|---|---|---|---|
| `--layer-split K` or `auto` | one GPU; setup places an `auto` split | contiguous layer ranges, one expert cache per card | setup's own text says prompts read about 20% faster; two RTX A4000 made unequal: prompt 945 to 1,441 tokens/s (+52%), decode 50.2 to 65.9 (+31%), 5/5 pairs |
| `--pipeline-windows N` | `-1` (auto): on for two or more gfx12 stages, opt-in on NVIDIA | the first card starts the next window while the last card verifies this one | RTX 4060 Ti + RTX 5080: mean decode 72.4 to 84.0 tokens/s (+16%); two RTX 3090: prefill 1,798 to 2,097 tokens/s, decode 137.0 to 136.7; four RX 7900 XT: 59.4 to 67.6 tokens/s at 4K |
| `STRATA_SPLIT_RING N` | `0` (the normal ring rule) | the streamed expert ring, in slots | two V100: 192 slots gave 5.39% faster 4K prefill than 384; 96 reduced 32K prefill |
| `STRATA_SPLIT_OVERLAP` | off | a ready flag instead of a host sync at the hand-off | two RX 7900 XTX: 53 to 70.1 tokens/s on a code prompt |
| `STRATA_SPLIT_OWN` | off | each card keeps its own prompt buffers instead of borrowing cache slots | R9700 + RX 9070 XT: 2K 1,450 and 16K 2,227 tokens/s; the answers differ |
| `STRATA_SPLIT_CHUNK16K` | off: a gfx12 split keeps 8192 | keeps the 16,384-token chunk on a split | 2x / 3x / 4x R9700 at 16K: +36% / +45% / +53% |
| `STRATA_SPLIT_SPEED_SCALE`, `STRATA_SPLIT_UNIFORM` | unset | placement inputs for `--layer-split auto` | a card priced at 0.57 of the other's speed moved the split from K=3 to K=20, which is the +52% prompt row above |
| the draft layer on the first card with the pipeline | default when `--pipeline-windows 2` runs; `STRATA_MTP_STAGE0=0` keeps the last card | the draft chain runs on a card whose window has finished | two R9700: decode 103 to 125 tokens/s on a code prompt (+21%), 4K prompt +8%, prose equal |
| `STRATA_PREFILL_HELP=1` | off | the idle card computes a share of the streamed experts of a one-chunk prompt; native MMQ packs only, and a stage without its own H buffer refuses the prompt | two RTX 3090: 1.1K 496 to 723 tokens/s, 2K 878 to 1,128, 3K 1,260 to 1,440; 4K and 8K unchanged |

Limits, card order and the split rules: [MULTI_GPU.md](MULTI_GPU.md).

### Vision

Hardware is named in the measured cell of each row.

| Setting | Default | Effect | Measured |
|---|---|---|---|
| `--vision` | off | loads the image encoder and keeps its VRAM free | Q2_0 at 4K: 90.7 to 86.9 tokens/s decode, 541 to 531 tokens/s prompt; 2-8% slower |
| the CPU encoder | default on an NVIDIA card under 24 GB | takes no VRAM from the expert cache | Tesla P100: 31.1 to 32.6 tokens/s; RTX A4000: 46.0 to 48.4 (decode, 65K context) |
| `"vision": {"lazy": true}` | off | the encoder starts only when a picture arrives, so its VRAM stays with the expert caches; not with batch slots | the encoder's room is about 850 expert slots of a 16 GB card's ~4,500 ([DETAILS.md](DETAILS.md#images-vision)) |
| `"vision": {"elastic": true}` | off | frees the encoder after 120 s idle; turns `--kv-grow` off | Tesla P100 and RTX A4000: 2.7 s a picture once loaded, 4.5-5.8 s for the first |

The image path itself: [DETAILS.md](DETAILS.md#images-vision). Image generation is a
different feature: [IMAGES.md](IMAGES.md).

## The optimization catalog

Each row names the switch that turns the item off or on, its status, and its measurement.
Where a whole page exists, the page holds the detail.

### Per-architecture default tables

The engine sets these before it reads the user's switches. A switch that the user sets is
kept, whatever the value. The tables live in `src/core/nv_defaults.cpp` and
`src/core/arch_defaults.cpp`.

Hardware is named in the measurement cell of each row.

| Table | Contents | Status | Measured |
|---|---|---|---|
| sm_86 (`STRATA_SM86_PERF=0` turns it off) | nine exact decode switches: `ADAPT_NOWAIT`, `ADAPT_LAG`, `CPU_IQ_ALL_EXACT1`, `SIDE_MIXER`, `ATTN_DAILY`, `GR_UP75`, `GU_SWIGLU`, `QSA_ROWS`, `B6_MMVQ_ROWS` | default on | 8 pairs, IQ3_XXS: RTX 3060 60.36 to 66.52 tokens/s (+10.2%), RTX A4000 51.16 to 57.82 (+13.0%) |
| sm_60 (`STRATA_SM60_PERF=0`) | seven switches: `ADAPT_NOWAIT`, `ADAPT_LAG`, `SIDE_MIXER`, `FUSE_WAIT`, `B6_MMVQ_ROWS`, `FUSE_DECODE`, `GR_UP75` | default on | Tesla P100 36.56 to 41.76 tokens/s (+14.2%), 8/8 pairs |
| sm_70 (`STRATA_SM70_PERF=0`) | the Volta master switch `SM70_TABLE`, `DEQUANT_WIDE`, `ADAPT_OVERLAP` (Linux), `PF_WMMA2`, and eleven exact decode switches | default on | fork's 2x V100 16 GB: 83.1 against 74.6 tokens/s (+11.4%), A/B/A/B/A/B with a restart before each request |
| sm_120 (`STRATA_SM120_PERF=0`) | `SIDE_MIXER`, `ATTN_DAILY`, `GR_UP75`, `GR_DOWN_TG`, `GU_SWIGLU` | default on | RTX PRO 5000 Blackwell: decode +13.8% / +15.8% / +13.6% (story / code / document, 6/6 pairs); prompt +9.9% / +5.4% / +3.4% at 4K / 20K / 98K |
| gfx12, RDNA4 (`STRATA_GFX12_DEFAULTS=0`) | `GDN_HEAD`, `GDN_PP`, `GDN_CONVL2`, `GDN_NOY`, `CVEC_FUSE` | default on | answers identical on an R9700; the decode half of the gfx1151 table measured slower there and stays out |
| gfx1151, Strix Halo (`STRATA_GFX1151_DEFAULTS=0`) | the prompt and decode tables, each entry with its measured gain in the source comment | default on | e.g. `GDN_PP=2` +2.9% on a 64K prompt, `Q8_PACKED` +2.3% decode |

`STRATA_DEQUANT_WIDE` is on by default on sm_86 and sm_61, and in the sm_70 table: the
time to the first token of a 150 / 300 / 450-token prompt fell 11-15% on the RTX 3060 and
6-10% on the RTX A4000.

### Prompt experts

Hardware is named in the measurement cell of each row.

| Item | Switch | Status | Measurement |
|---|---|---|---|
| FP16 tensor-core prompt experts | `STRATA_PF_WMMA=1` | opt-in, CUDA 12 Volta build | V100-SXM2 32 GB: +10% prompt speed against the MMQ `dp4a` kernels; the activations are FP16, so the answers differ |
| Second prompt-expert tile (128 x 64) | `STRATA_PF_WMMA2`, `STRATA_PF_WMMA2_N64` | on in the sm_70 table | contributor's +2.5-4.3% prompt speed, V100 |
| Lean prompt buffers | `STRATA_PF_WMMA_LEAN=1` | opt-in | two V100: 8K 5,258.6 to 5,082.9 ms (+3.5%), 32K 14,340.9 to 14,066.4 ms (+2.0%); 2K decode was slower in two samples |
| Shared region for the hyper-connection read | `STRATA_PF_HC_REGION=1` | opt-in | no isolated number; it is part of the same combined port as the group-scratch row |
| Group scratch on an all-MMQ prompt plan | `STRATA_PF_GROUP_SCRATCH=1` with `STRATA_PF_WMMA=0` | opt-in | two V100, two A-B-A-B pairs: 2K prompt 2,360.8 to 2,214.3 ms, 8K 6,932.7 to 6,541.0 ms; planner scratch at 8,192 tokens 3.618622 to 2.587509 GiB, 2,976 to 2,128 borrowed slots |
| Pair trellis port: codebook in shared memory, eight-lane rows | `STRATA_TQ_SUB8=1` with `STRATA_EXPERT_PAIR=0` | opt-in | two V100: one expert call 242.5 to 153.3 us (card 0) and 242.1 to 154.7 us (card 1); long-reply decode 120.2 to 126.4 tokens/s, indicative |
| Fused int8 prompt experts | `STRATA_PF_FUSED` | on for sm_120 with 80 or more SMs and for gfx12; off elsewhere | RTX PRO 5000: prompt +9.9% at 4K; R9700: 4K +15.7%, 32K +19-20%, a 96K prompt +16% |
| MMQ reads resident experts in place | `STRATA_MMQ_DIRECT=1` | opt-in, needs `-DSTRATA_MMQ_TAB=ON` | 4x RTX 3090, every expert resident: a 170-token prompt 331 to 251 ms, 1,217 tokens 773 to 634 ms |
| Loan sized to the actual read | `STRATA_PREFILL_LOAN_EXACT=1` | opt-in | two V100: a 6,143-token prompt 4.773 to 4.212 s (+13.3%) |
| Embedding buffer accounting | `STRATA_EMB_REUSE_ACCOUNT=1` | opt-in | 320 MiB a chunk returned to the planner; 3.9% faster prompt on an RTX 3060 |

### Decode kernels

Hardware is named in the measurement cell of each row.

| Item | Switch | Status | Measurement |
|---|---|---|---|
| Volta master switch | `STRATA_SM70_TABLE` | on in the sm_70 table | GPU work of one verify window 22.8 to 21.2 ms (V100-SXM2) |
| Wide dequantization | `STRATA_DEQUANT_WIDE` | default on for sm_61 and sm_86; on in the sm_70 table | bit exact; first-token time -11-15% (RTX 3060), -6-10% (RTX A4000) |
| Daily decode attention | `STRATA_ATTN_DAILY=1` | default on (sm_86, sm_70, sm_120) | +3.2% (5/5) on the RTX 3060, +0.7% on the RTX A4000 |
| Verify-window block scores with the queries in registers | `STRATA_QSA_ROWS=1` | default on (sm_86, sm_70); opt-in elsewhere | RTX 3060 + RTX 2080 Ti split: -0.09 ms a window at T=1, -0.37 ms at T=4; end to end +3.2% and +0.1%; -0.3% on an RTX 3060 alone ([QSAROWS.md](QSAROWS.md)) |
| Verify-window experts in shared memory | `STRATA_EXPDEC=1` | default on in the sm_70 table; opt-in elsewhere | gate/up -11% to -23% on the RTX 2080 Ti; end to end +1.1% on a 131K decode; `STRATA_EXPDEC_DOWN=1` was about 10% slower and stays off ([EXPERTDEC.md](EXPERTDEC.md)) |
| Verify-window expert cap | `STRATA_VERIFY_ECAP=N` | opt-in, changes the answers | 32K decode +17.7% at `N=20` (two V100) |
| Gate/up, SwiGLU and q8_1 in one launch | `STRATA_GU_SWIGLU=1` | default on (sm_86, sm_70, sm_120) | a fixed 4-token round 13.70 to 11.80 ms on the RTX PRO 5000 |
| Interleaved small-batch matvec | `STRATA_MMVQ_IL`, `STRATA_MMVQ_IL_WIDE`, `STRATA_MMVQ_IL_120`, `STRATA_MMVQ_IL_ROWS` | default on for sm_80 and newer; on in the sm_70 table; off on sm_75 and older | RTX PRO 5000: Q5_K head at 5 tokens 899 to 491 us; the 5-8 token table is sm_120 only ([MMVQ_IL_TABLE.md](MMVQ_IL_TABLE.md)) |
| Resident-biased routing | `STRATA_ROUTE_RESIDENT`, `_RANKS`, `_MTP` | opt-in, changes the answers | no end-to-end number in this repository; the route counters print at exit |

### Memory and copies

Hardware is named in the measurement cell of each row.

| Item | Switch | Status | Measurement |
|---|---|---|---|
| KV upload on a second stream | `STRATA_KV_PREFETCH=1` | opt-in | two RTX 3080 20 GB: prefill -5.3% at 32K, -4.1% at 64K; no decode gain ([KV_PREFETCH.md](KV_PREFETCH.md)) |
| Host-copy DMA for the K/V append | `STRATA_KV_HOST_DMA=1` | opt-in | two V100: one layer's 8,192-token INT8 append 9.345 to 2.875 ms (x4 link) and 2.027 to 0.896 ms (x16); no consistent whole-request gain |
| Batched DMA submission | `STRATA_DMA_BATCH=1` (`=2` adds the overlap hint) | experimental, needs `--pcie-mode dma` and CUDA 13 | RTX 4070 Laptop: submission 0.0936 to 0.0155 ms (-82.7%), total transfer -1.3%; a full-model screen showed 30.3 against 30.0 tokens/s ([BATCHED_DMA.md](BATCHED_DMA.md)) |
| Exchange buffer rotation | `STRATA_EXCHANGE_ROTATE=1` | opt-in | RTX 5090: 163.4 to 187.4 tokens/s; RTX PRO 6000: 104.90 to 113.44 tokens/s, identical output ids ([EXCHANGE_ROTATION.md](EXCHANGE_ROTATION.md)) |
| Disjoint expert slots across a helper and the primary | `STRATA_DISJOINT_ADAPT=1` | opt-in | two RTX 3080 20 GB: 81.5 to 94.1 tokens/s (+15.5%) ([DISJOINT_EXPERT_CACHE.md](DISJOINT_EXPERT_CACHE.md)) |
| Pinned stage buffers | `STRATA_STAGE_PIN=1` | opt-in, off after a corruption report | Tesla P100 with the RAM capped at 24 GB: 13.9 to 15.0 tokens/s (+8%) |
| Host-callback warm-up | `STRATA_ADAPT_CALLBACK_WARMUP` | default on, Linux, since 0.1.43.1 | RTX 5090 with a Ryzen 9 9950X: 142.82 to 207.75 tokens/s |
| Adaptive copies one window later | `STRATA_ADAPT_LAG=2` | default on sm_60 and sm_86; opt-in elsewhere | Tesla P100 +3.5% (6/6); RTX 3060 -1.3% |
| Per-layer PCIe balance | `STRATA_PCIE_BALANCE=1` | opt-in, one GPU, no `--pcie-frac` | RTX A4000 arena: 48.45 to 51.15 tokens/s (+5.6%, 5/5); RTX 3060 -1.6%, Tesla P100 -1.3% |
| Compact CPU activation download | no switch (always on) | default | a layer that sends no expert to the CPU copies nothing; at 2,048 tokens the old copy was about 21 MB a layer |

### Attention selection and scoring

Hardware is named in the measurement cell of each row.

| Item | Switch | Status | Measurement |
|---|---|---|---|
| Exact compact top-k selection (1Cat-vLLM #1171) | `STRATA_TOPK_COMPACT=1` | opt-in | two V100: selection 0.117 to 0.082 ms at 151,309 cells, 0.302 to 0.215 at 524,288; batched decode 62.1 to 64.6 tokens/s at 4,247 |
| Exact tiled prompt scoring (1Cat-vLLM #1176) | `STRATA_SCORES_PROMPT_HALVE=1` | opt-in; the V100 profile's `STRATA_SELECT_SIMT=1` stays faster | V100: 256-query scorer 0.825 to 0.659 ms at 32K, 3.527 to 2.489 at 131K, 5.631 to 3.916 at 200K |
| Tiled SIMT prompt scorer | `STRATA_SELECT_SIMT=1` | on in the V100 profile | +12.2% on a 131K prompt (two V100) |
| Prompt top-k with one read of the scores | `STRATA_TOPK_SAMPLE` | part of `STRATA_SM75_PERF=1` | sm_75 report only, not repeated here |

### Conversation and prompt cache

Hardware is named in the measurement cell of each row.

| Item | Switch | Status | Measurement |
|---|---|---|---|
| Checkpoint at the previous turn boundary | `STRATA_CACHE_MESSAGE_BOUNDARY=1` | opt-in | two RTX 3080 20 GB: first edit 17.236 to 11.527 s; hot generation 97.55 against 97.60 tokens/s ([MESSAGE_BOUNDARY_CACHE.md](MESSAGE_BOUNDARY_CACHE.md)) |
| Extra checkpoint near the prompt's end | `--prompt-cache-tail` | opt-in | no number in this repository |
| Deferred input checkpoint | `STRATA_INPUT_ANCHOR_DEFER=0` selects the synchronous control | default on | two V100, nine tool turns: TTFT -67 to -134 ms, all nine medians improved; a 1,082-token read took 1.815 s ([v100-incremental-ttft-2026-10-21.md](../benchmarks/v100-incremental-ttft-2026-10-21.md)) |
| Checkpoint pool and copy threads | `STRATA_CKPT_POOL=1`, `STRATA_CKPT_THREADS=1` | on in the V100 profile | no isolated number |
| Retained disk root | no switch (always on) | default | repeated shared-root HTTP latency 1,799.7 to 902.6 ms (two V100) |
| Pinned shared document | `strata_prefix` in the request | opt-in per request | Tesla P100, 64K document: questions 2-5 48.6 to 0.35-0.53 s ([RESEARCH_RUNS.md](RESEARCH_RUNS.md)) |

### Speculation

Hardware is named in the measurement cell of each row.

| Item | Switch | Status | Measurement |
|---|---|---|---|
| Coupled and Gumbel draft sampling | `STRATA_SPEC_COUPLED=1`, `STRATA_SPEC_GUMBEL=1` | opt-in | RTX 3060: accepted drafts 62.1% to 66.9% and 68.0%; decode 43.3 and 42.8 tokens/s, inside the spread |
| Speculative rejection sampling | `STRATA_SPEC_PROB=1` | opt-in | no speed gain on 2x RTX 3090 (137.2 against 137.0 tokens/s) |

### CPU pool

Hardware is named in the measurement cell of each row.

| Item | Switch | Status | Measurement |
|---|---|---|---|
| Exact one-token expert rows | `STRATA_CPU_IQ_ALL_EXACT1=1` | default on sm_86 | IQ3_S row -32% on a Threadripper PRO 3975WX; end to end +0.5% (Xeon E5-2643 v3 + RTX A4000) |
| Spare CPUs for the helper threads | `--aux-cpus auto` | off | 2x RTX 3080 20 GB: solo decode 79.7 and 80.4 against 72.1 and 72.4 tokens/s |
| Pool spin | `STRATA_POOL_SPIN_US` | `100` | a zero-spin microbenchmark 139.8 to 46.2 ms of CPU time; no gain at `100` |
| AVX-2 one-token expert rows | `STRATA_IQ_MT_MIN=1` keeps the old rule | default on an AVX2 CPU with no AVX-512 | IQ3_S -32% on Zen 2, -11% on Zen 3, -60% on a VNNI core |

## Measured and rejected

Do not repeat these. Each row is a result measured in this repository; the second column
names the hardware.

| Setting | Hardware | Result |
|---|---|---|
| `--pcie-frac 0`, `--pool-workers 10`, `--adapt-swaps 24`, `--adapt-every 8`, `--adapt-every 0`, `CUDA_SCALE_LAUNCH_QUEUES=4x`, `CUDA_DEVICE_MAX_CONNECTIONS=32` | two Tesla V100-PCIE-16GB | inside the noise of two baseline arms; a 32K prompt 14,281 against 14,259 and 14,263 ms |
| the CPU/PCIe pool-rate rule of upstream `213f1a39` | two Tesla V100-PCIE-16GB | median decode 128.48 against 129.09 tokens/s; removed |
| dual data-parallel prefill (upstream #5002) | two Tesla V100-PCIE-16GB | a 32K chat check 18,749.3 ms off, 22,859.2 ms on |
| `STRATA_FS_SLOTS=4` | RTX 3060 12 GB, Tesla P100 16 GB, RTX 5070 | 0.95 (0/6), 0.97 (2/6), 0.98 (1/6); 16 slots on the P100 0.96 (0/6) |
| `STRATA_GDN_CHUNKED=2` (forced) | RTX 3060 12 GB, RTX 5070 | 0.73x and 0.87x the default's recurrence |
| `--prefill auto:16384` on a layer split | two Tesla V100-PCIE-16GB | 1,607 against 1,736 tokens/s on a 32K prompt (-7.5%) |
| `--prefill 4096` and `--prefill 2048` | two Tesla V100-PCIE-16GB | 23% and 54% slower on a 32K prompt |
| `STRATA_SPLIT_RING=384` and `512` | two Tesla V100-PCIE-16GB | no faster than 192 on a 32K prompt |
| `STRATA_VERIFY_ECAP=16` on an agentic loop | two Tesla V100-PCIE-16GB | 12 of 20 turns ended as truncated reasoning against 0 of 20; it also accepted more drafts (62.3% against 56.7%) |
| `STRATA_SIDE_MIXER=1` (short context) | RTX A4000 16 GB | -0.6% |
| `STRATA_HC_REQ8=1` | RTX 3060 12 GB | -0.2% decode |
| `STRATA_ADAPT_LAG=2` on its own (0.1.41 switch study) | RTX 3060 12 GB | -1.3% decode; the 0.1.45 sm_86 set keeps it because the set pays as a whole |
| `STRATA_ADAPT_OVERLAP=1` | RTX 5070 on Windows, RTX 5090 Laptop | -2.6% on Q2_0; -21% on the laptop (84.6 against 106.6 tokens/s) |
| `STRATA_PCIE_BALANCE=1` | RTX 3060 12 GB, Tesla P100 16 GB | -1.6% and -1.3% decode |
| `--batch-mtp` | two R9700 with every expert in VRAM | 80.2 to 62.7 tokens/s (-21.9%, 0/5 pairs) |
| `STRATA_PF_WMMA_LEAN=1` at 2K | two Tesla V100-PCIE-16GB | decode 80.0 / 118.6 against 101.5 / 122.8 tokens/s in the two samples |
| `STRATA_DMA_BATCH=1` as a model change | RTX 4070 Laptop 8 GB | 30.3 against 30.0 tokens/s; the submission gain did not reach the model |
| `STRATA_KV_HOST_DMA=1` as a model change | two Tesla V100-PCIE-16GB | no consistent prefill gain on 2,048 / 9,436 / 29,728-token HTTP prompts |
| `STRATA_ROUTE_TAIL_SKIP=7` above 90% cached | Tesla V100 32 GB, Coder IQ1_M, 94.8% cached | draft acceptance 70.1% to 68.2%, decode 1-2% slower than 0.1.41 |
| `STRATA_ROUTE_PRIOR=0.5` | RTX 3060 12 GB, RTX A4000 16 GB | teacher-forced KL 0.0126 against a floor of 0.0012; top-1 agreement 96.6% against 99.1%; HumanEval 154/164 against 157 and 159 |
| `--spec-split` | Arc A770 | 9.9 / 10.2 / 10.3 against 11.7 / 12.1 / 12.6 tokens/s |
| `--adapt-swaps 0` | Arc A770 | 10.5-11.2 against 12.0 tokens/s |
| `STRATA_SPEC_COUPLED=1` with `STRATA_SPEC_GUMBEL=1` | two RTX 3090 | 136.5 against 137.0 tokens/s |
| `STRATA_PREFILL_CPU_SHARE=0.30` and `0.75` | two Tesla V100-PCIE-16GB | slower at 1K tokens; the automatic share stays on |
| `STRATA_EXPDEC_DOWN=1` | RTX 2080 Ti + RTX 3060 split | about 10% slower than the default down kernel |

## The tune API

The tuner changes settings on the running server, measures them, and keeps a win. The
commands and their output are in [TERMINAL.md](TERMINAL.md); the API routes are in
[SERVER.md](SERVER.md).

### Three per-request settings

These need no restart. The request carries them in `sampling.strata_tune`
(`strata_control/tune_knobs.py`, `serve/server.py`). No hardware applies to this table.

| Key | Range | What it changes | Answer class |
|---|---|---|---|
| `pcie_frac` | 0.0 to 1.0 | the share of a missed expert set that the GPU reads over PCIe | placement |
| `spec_min_p` | 0.0 to 1.0 | the draft layer's confidence floor for the next draft | exact |
| `aux_cpus` | `1` or `0` | puts the engine's helper threads on the spare CPUs, or back | exact |

`prefill_pipe` is also forwarded, for the value that `--calibrate` measured. A request without
the key returns to the start-up setting. With `--batch`, the most recent request's value holds.

### The arm list

An arm is a single-knob change of the user's own configuration. It never starts from a
default grid.

| Rule | Effect |
|---|---|
| bounds | a value outside them is not tried: `--spec` 3-4, `--spec-min-p` 0.3-0.7, `--pcie-frac` 0.0-0.55, `--adapt-every` 1-4, `--adapt-swaps` 80-160, `--adapt-decay` 0.7-0.97, `--kv-resident` 32768 / 49152 / 65536, `--prefill` auto / auto:16384 / auto:32768 / 2048 / 4096 |
| the never list | a setting that measured worse on this machine class is skipped, with the reason in plain words, for example `--spec` 6 and above (4% to 16% slower) and `--spec` 2 on Flash-Next (4% to 11% slower) |
| machine class | A to G, from the card, the RAM, the backend and the model family; `deep_only` arms run last |
| needs | an arm runs only when its facts hold, for example `resident_ram`, `ctx_ge_128k` or `vram_ge_24g` |
| the measurement | one warm-up, three decodes of 200 tokens, one 4,096-token prompt; a win needs 3% or more in every round |
| order | every per-request setting is tried before any restart |

### Answer classes

| Class | Meaning | Notice |
|---|---|---|
| `exact` | answers identical to the standard setup | "Answers are identical to the standard setup." |
| `placement` | the same math with a different CPU/GPU split, so a near tie can flip | "Answers may differ slightly from the standard setup." |
| `nonrepeatable` | not bit-exact from run to run | "Answers are not repeatable run to run." |
| `quality` | changes the numbers on purpose; never picked automatically | "Includes a quality setting that changes the numbers." |

On a PC where the standard setup itself is not repeatable (the resident-RAM mode, or SYCL),
the tuner says so instead of promising identical answers.

## Rules for quoting numbers

1. Label the measurement. A component number is not a model number. `STRATA_EXPDEC=1` cut the
   gate/up kernel by 23% and the whole decode by 1.1% on the same two cards.
2. Do not present a synthetic prompt as everyday speed. The V100 tables use random-word prompts,
   and other tables use one fixed text. A real agent turn reads a different text and reuses a
   prefix.
3. Give the hardware, the engine version and the pair count with the number. Put the hardware in
   the table caption.
4. Report the median and the range of the samples. One run is a sample, not a result.
5. Do not move a number to other hardware. A switch that pays on an RTX A4000 can cost speed on
   an RTX 3060, and the reverse.
6. Mark a quality setting. `--kv q4_0`, `STRATA_VERIFY_ECAP`, `STRATA_ROUTE_PRIOR` and
   `STRATA_ROUTE_RESIDENT` change the answers or their distribution.
