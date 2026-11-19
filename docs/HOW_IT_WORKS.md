# How Strata runs a large model on your PC

Strata runs Qwen3.8-Flash-Next on a personal computer. The model files are 58 to
188 GB ([README](../README.md#models-and-quants-that-run-here)). A personal graphics
card holds 12 to 24 GB of VRAM. The engine closes that gap with three memory tiers and one
rule: a part that every token needs stays on the card, the experts move.

The comparison, once: the engine works like a kitchen. The tools for every dish stay on
the counter, the rest wait in the pantry, and a list says where each tool is. The counter
is the VRAM, the pantry is the RAM, and the list is the residency table.

This page gives the parts and their order. [DETAILS.md](DETAILS.md#how-it-works) holds all
measured numbers, and [paper/Strata-Paper.pdf](paper/Strata-Paper.pdf) the full study.
Start at the [documentation index](README.md).

## Who does what

<p align="center"><img src="media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **VRAM:** the parts that every token needs. These are the attention and GDN mixers,
  hyper-connection weights, routers, the shared expert, the output head, the draft layer and
  the KV cache. An **expert cache** fills the rest with the most-used routed experts.
- **RAM:** all 24,576 expert blobs, page-locked. The CPU computes the rows of the experts
  that no card holds.
- **SSD:** the n-gram (PLE) table, read a few small rows per token.
- The engine overlaps the tiers: the card runs its experts while the CPU runs the others.

The sections below give each part in technical terms.

## What the model is

The model is a mixture of experts. Every token passes all 48 trunk layers, and each layer
runs 10 of its 512 experts. The shape is fixed in the code, and a model file of another
shape is refused at load:

| Property | Value | Source |
|---|---|---|
| Trunk layers | 48: 36 gated delta-net (GDN) and 12 full attention (QSA) | `include/strata/core/geometry.hpp` |
| Full-attention layers | every 4th: 3, 7, ... 47 | same |
| Layer width | 2,560 values | same |
| Routed experts | 512 per layer, 10 per token, 24,576 in the trunk | same |
| Expert width | 640; the shared expert is 640 wide as well | same |
| Vocabulary | 248,320 tokens | same |
| QSA heads | 24 query heads, 2 KV heads, head width 256, rope on 64 values | same |
| QSA selection | the indexer picks the best 2,048 cells, one pooled key per 4 cells | same |
| GDN state | width 128, 16 key heads, 48 value heads, convolution width 4 | same |
| Residual | a stack of 4 streams (hyper-connections) through a rank-320 mix | same |
| Draft layer | one multi-token-prediction (MTP) block, after the trunk | `include/strata/core/mtp.hpp` |
| PLE rows | 16 rows per token, 160 values per row | `include/strata/kernels/ngram.hpp` |

Each block does the same steps. It reads the residual stack through the hyper-connection
mix and runs the mixer, GDN or QSA. It writes the stack back, routes 10 of 512 experts and
runs them with the shared expert. It adds both results to the stack.

The model does not fit in VRAM, and no card size makes it fit. The routed experts alone
are 34.8 GiB in UD-Q2_0 and 55.4 GiB in UD-IQ4_XS ([QUANTS.md](QUANTS.md)). The dense
part is about 4.8 GB, and the PLE table is 26.8 GiB. So the engine holds a part of the
experts and computes the rest elsewhere.

## The three memory tiers

### VRAM: the expert slots

The expert cache is a block of VRAM cut into slots. One slot holds one expert blob:
1,382,400 bytes in the IQ packs (`include/strata/core/expert_source.hpp`). A residency
table maps each (layer, expert) pair to a slot, or to `-1` when no slot holds it
(`include/strata/core/expert_cache.hpp`). One GB of VRAM therefore holds about 700 more
experts.

Before the cache takes the rest of the VRAM, the engine prices what must stay beside it.
The plan holds back the draft head, the prompt buffers when they cannot borrow, one slot
for the spec residency table, and the reserve (`--vram-reserve-mib`, 700 MiB). It prints
the plan. On two Tesla V100-16GB with a 524,288-token int8 KV and layer split 20, the plan
opened 8,255 slots on CUDA0 and 5,750 on CUDA1
([DETAILS.md](DETAILS.md#how-it-works)).

A slot is filled once and stays filled; the cache does not evict on its own. At start, a
profile fills the slots in routing order (`--expert-profile`). After that, only the
adaptive tier replaces an expert. `--expert-cache N` sets the slot count. `0` (the
default) allocates none, unless a profile is given: then `0` means `auto`, and setup's
configurations write `auto`. Without a profile, the cache fills slots in arrival order
from one counter, so the first positions take them all. `--expert-cache-per-layer` gives
each layer its own range.

### RAM: the pinned arena

Every expert blob is copied from the pack into one page-locked block of RAM at start
(`src/core/expert_source.cpp`). The CPU reads a blob in place and writes its result, with
no copy of the blob. The prompt path streams a chunk's blobs from the same arena to the
card over PCIe. The copies pass through a pinned ring on a copy stream
(`include/strata/prefill/prefill.hpp`).

The CPU expert pool computes the experts that no slot holds
(`src/kernels/cpu/pool.cpp`). It uses every physical core except the one the host loop
spins on (`--pool-workers 0`). The host thread joins the drain by default, and
`--no-host-worker` keeps it out. The pool works at the same time as the card. The host
sums both results at the end of each layer, so the layer is a barrier for the pool.

A missed expert can also be read by the card over PCIe. `--pcie-frac` sets the share of
the misses that the card reads: `0.2` for the Q2_0 pack and `0.55` for the native packs
(`src/program/generate.cpp`). A start-up probe measures the link, and `--serve` refines the
share from the first decode windows.

If the RAM cannot hold the experts, `--mmap-experts` keeps them in the pack file. The OS
file cache then holds what the card does not. The resident modes and their RAM costs are
in [MODELS.md](MODELS.md).

### SSD: the PLE table

The PLE is a layer-1 input. Sixteen row indices come from the token and its two
predecessors, with a 64-bit multiply and xor on the host
(`include/strata/kernels/ngram.hpp`). The 16 rows hold 160 values each and feed layer 1.

| Property | Value | Source |
|---|---|---|
| Rows | 320,001,536 | `include/strata/kernels/ngram.hpp` |
| Bytes per row | 90 (IQ4_NL); 160 (FP8); 170 (Q8_0) | same |
| Size | 28.8 GB (IQ4_NL); 51.2 GB (FP8) | same |
| Reads per token | 16, on 16 pages | `include/strata/ngram/ple_reader.hpp` |

The table is the only tensor the engine reads from the original model file instead of the
pack. `--ple-io direct` (the default) reads it unbuffered, so the table never enters RAM.
`--ple-io ram` locks the whole table in RAM, and `--ple-io mmap` is the A/B arm.
`--ple-row-cache` (1048576 rows) keeps whole 4 KiB pages that the process has read.

### The adaptive tier

The adaptive tier changes which experts the slots hold. It counts the routing of every
(layer, expert) pair while the engine answers. Every `--adapt-every` verify windows
(default 4) it compares the counts and swaps up to `--adapt-swaps` experts (default 96)
into the slots. A count decays by `--adapt-decay` (default 0.7) after each adaptation, so
old evidence fades. An expert enters only when its decayed count beats the count of the
expert it replaces by more than `--adapt-min-gain` (default 1.5). A higher value means
fewer, surer swaps.

A swap copies a blob from the RAM arena into a slot. The expert that leaves the slot
becomes a CPU miss. The swap changes which device computes that row, so a near-tie can
flip. `--expert-profile-save` writes what the tier learned, and a later start can load it
as the profile.

`--adapt-async 1` is opt-in. It advances the swaps between verify windows on a helper
thread, so no window waits for a whole round. The run is then not bit-exact from run to
run. `--adapt-every 0` keeps the slots fixed.

## Guess, then check

<p align="center"><img src="media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check.** The model's own draft layer proposes the next few tokens, and one
  pass of the trunk checks all proposals at once. The engine keeps the proposals that the
  trunk agrees with and adds the token the trunk picked itself.
- The draft layer decides nothing. Every emitted token is one the trunk picked. For a
greedy request, the answer is the same as a plain one-token-at-a-time run. With sampling,
the engine judges a draft against the token the request sampled.
- When the reply repeats the context (code edits, quoted text), **prompt lookup** adds
  drafts from the earlier copy. It does so only where its measured acceptance and cost say
  it pays.

In technical terms:

- A **verify window** holds T rows: the last accepted token and T-1 drafts. T is at most 8
  (`kVerifyMaxT`, `include/strata/kernels/verify_kernels.hpp`). Setup writes `--spec 4`,
  which is one token and three drafts.
- One pass runs the T rows through all layers of a stage, and the head gives the argmax of
  every row. The engine accepts the longest prefix where each draft equals that argmax,
  commits it, and emits those tokens (`src/core/verify.cpp`).
- The drafts come from the MTP block (`src/core/mtp.cpp`). `--spec-min-p` stops the window
  from extending when the draft layer is unsure. The suffix drafter adds up to 5 tokens from
  an earlier copy of the same text (`--suffix-draft`, default 3). `--lookup-chain`
  continues the MTP drafts from a prompt match.
- A pass returns more than one token: 2.4-3.2 tokens per pass on average
  ([DETAILS.md](DETAILS.md#how-it-works)). The engine's own per-window report read 2.49
  tokens per window with the expert cap off, on two Tesla V100-16GB with IQ3_S and
  `--spec 4` ([NVIDIA_V100.md](NVIDIA_V100.md#verify-window-expert-cap)).

## Reading long texts

**Long texts are read in big pieces** of up to 8,192 tokens, so a long document is read
at more than 1,000 tokens/s. One Tesla V100-PCIE-16GB read an 8K prompt at 1,442.4
tokens/s ([DETAILS.md](DETAILS.md#speed-measured)). After the first message, the engine
keeps the conversation and reads only what is new, so a follow-up starts in seconds.

In technical terms:

- `--prefill CHUNK` reads the prompt in chunks of CHUNK tokens. The native packs require
  it (`src/program/generate.cpp`). `auto` takes the largest chunk up to 8,192 tokens whose
  buffers the expert cache can lend; `auto:16384` and `auto:32768` allow bigger chunks.
- One chunk goes through all 48 layers (`src/prefill/prefill.cpp`). The projections are
  tensor-core products. The routed experts are grouped by expert. Resident experts are read
  from their slots, and the others stream from the RAM arena through a pinned ring.
- The prompt path borrows the last slots of the expert cache for its own buffers, then
  gives them back. When the cache cannot lend, the chunk is smaller.
- On a layer split, the chunks flow through the stages as a pipeline. `--prefill-pipe`
  sizes them for that flow ([MULTI_GPU.md](MULTI_GPU.md)).

## One request end to end

The engine takes token ids and returns token ids. It has no tokenizer
(`src/program/generate.cpp`). The server does the text work.

1. **Chat template and tokenizer.** The server turns the request into the model's message
   form, renders the Jinja template, and tokenizes the text
   (`serve/frontend.py`, `tools/strata_tokenizer.py`).
2. **Request line.** The server writes one line to the engine's stdin:
   `GEN <max_new> [sampling keys] <id,id,...>` (`serve/server.py`, `StrataEngine`).
   A request with images uses `GENI` and an embeddings file.
3. **Prefix reuse.** The engine compares the prompt with the checkpoints it holds. On a
   match it restores the running state and reads only the tokens after the match
   (`src/core/conversation_*.cpp`).
4. **Prompt read.** The engine reads the new tokens in chunks through the prompt path
   (`src/prefill/prefill.cpp`). Each chunk leaves the session state where the token path
   would have left it.
5. **Draft.** The MTP block proposes the tokens of the next window and appends its own K/V
   (`src/core/mtp.cpp`). The suffix drafter can add drafts from a repeated span
   (`src/spec/suffix_drafter.cpp`).
6. **Verify window.** The window's rows run through the layers (`src/core/verify.cpp`).
   Resident experts come from the slots and the others from the CPU pool. The head gives
   one argmax per row.
7. **Accept and sample.** The engine keeps the matching prefix. Greedy is the default; with
   `--seed` the head samples with top-k, top-p and temperature, outside the captured graph
   (`src/kernels/cuda/sampler.cu`). Each accepted token goes to stdout as `T <id>`.
8. **Detokenize and stream.** The server decodes each token incrementally
   (`serve/server.py`, `Detokenizer`). It holds the bytes of a character that is split
   across tokens until the character is complete, then streams the text.

The engine ends the answer with one `DONE` line that carries the counters: prompt tokens,
prompt time, decode time, drafts offered, drafts accepted. The server turns those into the
API usage block and the Monitor tab.

## The KV cache and the conversation cache

### What the state holds

- The 12 QSA layers keep a **KV cache**: one key and one value per past position, for
  each of the 2 KV heads (`include/strata/core/layer.hpp`). A **cell** is one position's K
  and V in one layer. `--kv fp16` is the default; `--kv int8` halves the bytes of a cell
  and `--kv q4_0` halves them again.
- The 36 GDN layers keep no KV cache. They keep a recurrent state and a convolution
  history per layer.
- The draft layer keeps its own K/V for the last `--mtp-window` cells (default 32,768).
- The session also holds the PLE history, which the hash of the next token reads.

`--kv-resident N` streams the KV cache. N cells of each QSA layer stay in VRAM and the
whole K/V lives in pinned RAM. The freed VRAM goes to expert slots. `0` (the default) keeps
all of it in VRAM. [ENGINE_FLAGS.md](ENGINE_FLAGS.md) lists the KV formats and their bytes
per cell.

### The three tiers of the conversation cache

| Tier | Where | Switch | Default |
|---|---|---|---|
| L1 | the live session, in VRAM | always | on |
| L2 | parked conversations, in host RAM | `--conversation-cache-mib` | `0` (off) |
| L3 | parked conversations, on disk | `--conversation-cache-disk` and `--conversation-cache-disk-gib` (both) | off |

An outgoing conversation parks in L2 first. A successful L2 park does not write to disk.
When L2 needs room, the engine moves a victim to L3 without a second read from the card.
L2-only entries do not survive a restart; L3 records do. The slot limit is
`--conversation-cache-slots` (default 4), and the RAM floor is
`--conversation-cache-min-free-mib` (default 2560). The shared system-prompt prefix gets
its own pinned entry, so its bytes are written once (keep-root, on by default). The full
policy, with the eviction and spill rules, is in [DETAILS.md](DETAILS.md).

### Checkpoints

A checkpoint is a copy of the running state at one token position
(`src/core/conversation_state.cpp`). It holds the GDN state and the PLE history. For each
QSA layer it holds the indexer's key tail, its spare rows and its block positions. It also
holds the token ids and the image keys. One state takes about 118 MB of host RAM.

`--prompt-cache N` (default 6) bounds the checkpoints of the live conversation, and
`--prompt-cache-every N` (default 16,384) adds a checkpoint inside a long prompt. On a
layer split, a checkpoint holds one part per stage.

A restore copies the state back into the session, and the read continues from that
position. A checkpoint holds no K/V: the cells of later positions are overwritten when
those positions run again. A parked conversation adds one K/V image per QSA layer, so an
L2 or L3 restore brings the cache with it. The engine picks the deepest checkpoint whose
token ids and images are a prefix of the request. Across the live session, L2, L3 and the
idle batch slots, the longest valid prefix wins. A disk record is used only when its prefix
is longer than every resident one. The engine validates every layer before it applies a
state, and a failed check refuses the record instead of continuing with half of it.

## Multi-GPU

A layer split puts one model on several cards of one make. The layers go to the cards in
contiguous ranges. The first card runs layers 0 to K-1, and the next card runs K onward.
The last card also runs the output head and the draft layer
([MULTI_GPU.md](MULTI_GPU.md)).

- **A stage** is one card's range. Each stage keeps its own session state: its layers' KV
  and GDN state. It keeps its own prompt buffers and an expert cache for its own layers
  only.
- **Host RAM** holds the expert arena once, shared by all cards. The CPU pool computes
  whatever no card holds.
- **What crosses the link:** per window, the stage hand-off. It carries the residual stack,
  the last layer's pending write and its injection: `hc * n_embd + n_embd + hc` floats per
  row, which is 12,804 floats, about 51 KB (`Verifier::handoff_floats`). A window of eight
  rows is a few hundred KB. Prompt chunks also cross it: each stage reads its missed
  experts from the arena over its own PCIe link.
- **The pipeline:** with `--pipeline-windows 2`, the first card runs the next verify window
  while the last card still checks the current one. The next window is a guess, and the card
  puts its state back when the guess is wrong ([MULTI_GPU.md](MULTI_GPU.md)).
  `1` overlaps only the short prompt reads that go through verify windows.
- **The split rule:** `--layer-split` takes the first layer of each later card, for example
  `18` or `16,32`. `auto` prices one decode window for every placement and keeps the
  cheapest; the faster card goes last, and each card gets at least 3 layers. A manual split
  keeps the order you wrote.

A second card mainly adds room for experts. On two Tesla V100-16GB with Q2_0 and
`--layer-split 20`, decode ran at 55-58 tokens/s against 19-22 on one card alone. The two
caches hold 93-98% of the routed experts against 73% for one card
([MULTI_GPU.md](MULTI_GPU.md#measured)).

## The server layer

The server is `serve/server.py`. It keeps one `strata serve` process per model resident,
and the weights, the arena and the expert slots load once. It starts the process and waits
for `READY`. After a crash, the next request starts a new process with the same command.
`--lazy` starts the HTTP API without the model. The first generation request loads the
model.

- **The pipe:** the server writes command lines to the engine's stdin and reads the
  engine's stdout. The commands are `GEN` and `BGEN` (generate), `STOP` (cancel), `SAVE`
  and `RESTORE` (session files), `VRAM` (give VRAM back between requests) and `QUIT`. The
  engine answers `T <id>`, `PP` (prompt progress), and `DONE`.
- **The request queue:** one inference lock (a FIFO) serializes requests, because one
  sequence is resident. HTTP admission reserves a place before the request waits, so a full
  queue refuses early instead of hanging.
- **Slots:** `--batch N` (2 to 8) gives the engine N batch slots, each with its own session,
  and requests then decode together ([BATCHING.md](BATCHING.md)). Each slot takes `BGEN`.
  The engine carves the slot sessions at start and falls back to fewer slots when they do
  not fit.
- **Streaming:** a streaming request answers as Server-Sent Events
  (`text/event-stream`), one JSON object per delta. The server holds back the text tail that
  could still become a stop string. A stop string split across tokens is then found and
  never sent.
- **The web page:** `serve/web/index.html` is served at `/`, next to `app.js`, the styles
  and the Monitor page. The page uses the same API as any other client.

## Where to read the code

| Area | File |
|---|---|
| Entry point, command line, decode loop | `src/program/generate.cpp` |
| Model shape and its checks | `include/strata/core/geometry.hpp` |
| One token through all layers | `include/strata/core/session.hpp`, `src/core/session.cpp` |
| One layer: mixers, router, expert combine | `src/core/layer.cpp` |
| Prompt-batched read | `src/prefill/prefill.cpp` |
| Verify window | `src/core/verify.cpp` |
| MTP draft layer | `src/core/mtp.cpp` |
| Draft sources (prompt lookup) | `src/spec/suffix_drafter.cpp`, `src/spec/draft_policy.cpp` |
| Sampler | `src/kernels/cuda/sampler.cu`, `include/strata/kernels/sampler.hpp` |
| Expert cache (VRAM slots) | `src/core/expert_cache.cpp` |
| CPU expert pool | `src/kernels/cpu/pool.cpp` |
| Host arena and expert reads | `src/core/expert_source.cpp` |
| PLE hash and table read | `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp` |
| KV cache and its formats | `include/strata/core/layer.hpp`, `include/strata/kernels/kv_stream.hpp` |
| Conversation cache and session files | `src/core/conversation_*.cpp` |
| API server, tokenizer, detokenizer | `serve/server.py`, `tools/strata_tokenizer.py` |
| Chat template and stream parsing | `serve/frontend.py` |
| Terminal | `strata.py`, `strata_cli/cli.py` |

The measurements on this page are in [DETAILS.md](DETAILS.md#how-it-works). The paper,
with the bottleneck analysis and the measurements:
[paper/Strata-Paper.pdf](paper/Strata-Paper.pdf). Back to the [README](../README.md).

## Credits

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen
  team; compressed versions by
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by
  UkisAI; the experimental
  [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) by Unsloth (its
  support follows [eddoursul/Strata](https://github.com/eddoursul/Strata)). Their licenses
  apply to the model files.
- Built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).
  Ideas from [Splash](https://github.com/incoai/splash),
  [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen). More in the
  [details](DETAILS.md#credits-and-licenses).
- The AMD (HIP) backend: [AMD_HIP.md](AMD_HIP.md) names the contributors and the machines
  it was validated on.

## License

Strata is open source under the [MIT License](../LICENSE). A few parts carry their own licenses: `engine/third_party/ggml`
(MIT, llama.cpp / ggml), the web app's font (SIL Open Font License 1.1) and the experimental speed projection's
vector in `terminal/data/experimental-speed-projection` (Qwen Community License 1.0, from the model's activations). The
models are not part of this repository; each model's own license applies to its files.
