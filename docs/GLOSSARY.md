# Glossary

The technical terms that these pages use, in alphabetical order. Each entry gives a
one-line definition and the page that explains it. [STYLE.md](STYLE.md) fixes the meaning
of `engine`, `server`, `pack`, `shard`, `stage`, `slot`, `expert`, `verify window`,
`draft layer`, `prefill` and `decode`; this page repeats those definitions and defines the
other terms.

A setting is written `--flag value`, and an environment variable `NAME=value`. A number
belongs to the hardware named beside it.

| Term | Meaning |
|---|---|
| **adaptive tier** | The part of the engine that moves experts between the VRAM slots and the RAM arena by counted routing. [TUNING.md](TUNING.md) |
| **answer class** | The tuner's label for how a setting changes the answer: `exact`, `placement`, `nonrepeatable` or `quality`. [TUNING.md](TUNING.md) |
| **arena** | One large memory block reserved once. The pinned host arena holds every expert; the VRAM arena is cut into expert slots. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#the-three-memory-tiers) |
| **cell** | One context position's key and value in one QSA layer; the KV cache is counted in cells per layer. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-state-holds) |
| **checkpoint** | A copy of a conversation's running state at one token position, kept in host RAM; one takes about 118 MB. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#checkpoints) |
| **conversation cache (L1/L2/L3)** | The three places a conversation waits between requests: L1 in VRAM, L2 in host RAM, L3 on disk. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#the-three-tiers-of-the-conversation-cache) |
| **CPU pool** | The worker threads that compute the rows of every expert that no VRAM slot holds. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#ram-the-pinned-arena) |
| **decode** | The generation of one token at a time; each step is one verify window. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#one-request-end-to-end) |
| **draft layer** | The model's MTP block, which proposes the tokens of a verify window and decides nothing. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#guess-then-check) |
| **engine** | The `strata` binary built from `src/` and `include/`. [ENGINE_FLAGS.md](ENGINE_FLAGS.md) |
| **expert** | One routed feed-forward sub-network of a mixture-of-experts layer; the model has 512 per layer. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-model-is) |
| **GDN** | Gated delta-net: the model's linear-attention layer, 36 of the 48; it keeps a recurrent state, not a KV cache. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-model-is) |
| **hyper-connection (GR)** | The gated residual: the residual is 4 streams of 2,560 values, mixed through a rank-320 matrix. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-model-is) |
| **IQ3_XXS** | An i-quant of llama.cpp's family (2-3.5 bit), used for the routed experts of several packs. [QUANTS.md](QUANTS.md) |
| **KV cache** | The keys and values of past positions that the 12 QSA layers re-read; the GDN layers keep a recurrent state instead. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-state-holds) |
| **layer split** | One model on several GPUs, each with a contiguous range of layers. [MULTI_GPU.md](MULTI_GPU.md) |
| **logits head** | The output projection that turns the final residual into one score per vocabulary token (248,320). [HOW_IT_WORKS.md](HOW_IT_WORKS.md#one-request-end-to-end) |
| **mixture of experts (MoE)** | A layer whose feed-forward is a set of experts, of which a router picks a few per token. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-model-is) |
| **MTP** | Multi-token prediction: the model's own draft layer, trained to guess the tokens after a position. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#guess-then-check) |
| **n-gram table** | The 320,001,536-row table that the PLE reads; 28.8 GB in IQ4_NL, on disk by default. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#ssd-the-ple-table) |
| **NVFP4** | A 4-bit float format: 64-value blocks of E2M1 codes with four UE4M3 scales. [NVFP4.md](NVFP4.md) |
| **pack** | The converted expert file set that the engine reads (`--pack`). [MODELS.md](MODELS.md) |
| **PCIe share** | The share of the missed experts that a card reads over PCIe instead of the CPU computing them (`--pcie-frac`). [TUNING.md](TUNING.md) |
| **pipeline window** | The overlap where the first card runs the next verify window while the last card verifies the current one. [MULTI_GPU.md](MULTI_GPU.md) |
| **PLE** | Per-layer embeddings: the layer-1 input that 16 rows of the n-gram table build for each token. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#ssd-the-ple-table) |
| **prefill** | The batched read of a prompt, in chunks, through all layers. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#reading-long-texts) |
| **profile (expert)** | A file of the routing frequency of every (layer, expert) pair, used to fill the VRAM slots. [ENGINE_FLAGS.md](ENGINE_FLAGS.md) |
| **projector** | The small network that maps the vision encoder's output into the language model's width. [DETAILS.md](DETAILS.md#images-vision) |
| **prompt cache** | The checkpoints that the engine keeps for the live conversation (`--prompt-cache`, default 6). [HOW_IT_WORKS.md](HOW_IT_WORKS.md#checkpoints) |
| **Q4_K** | A 4-bit K-quant of llama.cpp's family; Unsloth's UD-Q4_K_XL uses it for the expert gate and up. [QUANTS.md](QUANTS.md) |
| **Q5_0** | A 5-bit block quant of ggml; Unsloth's v3.1 sets use it in some down projections. [QUANTS.md](QUANTS.md) |
| **QSA** | The model's full-attention layer, 12 of the 48; an indexer picks the 2,048 cells that the attention reads. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-model-is) |
| **quantization** | A stored number format for weights: a block of values with a scale. [QUANTS.md](QUANTS.md) |
| **ring (pinned ring)** | The fixed set of expert blobs that the prompt path streams from RAM to a card in turn, on a copy stream. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#ram-the-pinned-arena) |
| **rope** | The rotary position encoding applied to a slice of each attention head (64 of 256 values here). [ENGINE_FLAGS.md](ENGINE_FLAGS.md) |
| **router** | The projection that scores all 512 experts of a layer and picks the 10 that a token uses. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-model-is) |
| **run config** | The `strata-<model>.json` file that setup writes and the server reads at start. [SERVER.md](SERVER.md) |
| **server** | `serve/server.py`, which starts the engine and serves the API. [SERVER.md](SERVER.md) |
| **serving profile** | A small JSON file of your own that overrides serving keys of one run config, such as `port` and `parallel`. [TERMINAL.md](TERMINAL.md) |
| **shard** | One GGUF file of a model that is split across files; the head and the PLE table have their own shards. [MODELS.md](MODELS.md) |
| **shared expert** | The dense feed-forward network that every token uses in every layer, beside the 10 routed experts. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#what-the-model-is) |
| **slot (batch)** | One request position of `--batch`; each slot decodes its own conversation with its own session. [BATCHING.md](BATCHING.md) |
| **slot (expert cache)** | One position in the VRAM expert arena that holds one expert blob; the residency table maps (layer, expert) to a slot. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#vram-the-expert-slots) |
| **stage** | One GPU's contiguous layer range in a layer split. [MULTI_GPU.md](MULTI_GPU.md) |
| **terminal** | The `strata` command in `strata.py` and `strata_cli/`, which starts and controls the server. [TERMINAL.md](TERMINAL.md) |
| **TQ2_0** | A trellis-coded quant type of agentionai's Gyro family (Gyro-M), in a Hadamard-rotated basis. [GYRO.md](GYRO.md) |
| **tune arm** | One candidate that the tuner measures: the user's configuration with one knob changed. [TUNING.md](TUNING.md) |
| **verify window** | The group of rows that one pass through the layers checks: the last accepted token and the drafts, at most 8. [HOW_IT_WORKS.md](HOW_IT_WORKS.md#guess-then-check) |
| **vision encoder** | The 27-layer ViT in the mmproj file that turns a picture into image tokens; `strata-vision` runs it. [DETAILS.md](DETAILS.md#images-vision) |
| **VRAM reserve** | The VRAM that a card keeps free for other programs (`--vram-reserve-mib`, default 700). [ENGINE_FLAGS.md](ENGINE_FLAGS.md) |
| **yarn** | A rope scaling method that extends the context past the trained length (`--rope-scaling yarn`). [ENGINE_FLAGS.md](ENGINE_FLAGS.md) |
