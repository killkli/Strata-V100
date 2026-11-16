# The API server

`serve/server.py` starts the engine, holds the model in memory and serves the HTTP API. It speaks
three dialects on the same port: OpenAI Chat Completions, Anthropic Messages and OpenAI Responses.
The web page at `/` is served by the same process.

The server reads one run config file, applies the command line on top of it, and starts the engine
as a child process. All facts on this page come from `serve/server.py`, `serve/runconfig.py`,
`serve/mcp.py` and `strata_control/tune_knobs.py` unless another source is named.

Related pages:

- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) - the process layout and the engine protocol.
- [TERMINAL.md](TERMINAL.md) - the `strata` command that starts this server.
- [ENGINE_FLAGS.md](ENGINE_FLAGS.md) - the engine's own options (`args` in the run config).
- [ENVIRONMENT.md](ENVIRONMENT.md) - the engine's environment variables.
- [TUNING.md](TUNING.md) - measured performance settings.
- [GLOSSARY.md](GLOSSARY.md) - the terms used here.
- [DETAILS.md](DETAILS.md#using-it) - request examples for every endpoint.
- [DETAILS.md](DETAILS.md#json-response-formats) - structured output and the JSON response formats.
- [BATCHING.md](BATCHING.md), [MULTI_GPU.md](MULTI_GPU.md), [VRAM_ELASTIC.md](VRAM_ELASTIC.md),
  [IMAGES.md](IMAGES.md), [MCP_SERVER.md](MCP_SERVER.md), [SAFETY.md](SAFETY.md),
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Start the server

The normal start is the terminal, which writes the command line for you:

```text
strata serve --config strata-qwen3.8-flash-next.json
```

The server alone starts with the module form:

```text
python -m serve.server --engine strata --config strata-<model>.json
```

`--engine strata` needs `--config`. `--engine mock` is the default of the module and needs no
config; it is a scripted engine for client tests, with no GPU and no model.

Two checks stop the start before any GPU work: an invalid option stops with the argparse usage
message, and a non-loopback `--host` without an API key stops with exit code 2. A bad run config
also stops the start, with the reason on standard error.

## Command line

`serve/server.py` declares 30 options (`serve/server.py:8323-8390`). A value given on the command
line wins over the same key in the run config, unless the row says otherwise.

| Option | Value | Default | Description |
|---|---|---|---|
| `--engine` | `mock` or `strata` | `mock` | `strata` runs the engine named by the config; `mock` runs a scripted engine with no GPU. The terminal passes `strata`. |
| `--config` | path | none | The run config (JSON). Required with `--engine strata`. |
| `--host` | address | none, then the config's `host`, then `127.0.0.1` | The address to listen on. `0.0.0.0` also answers other devices and needs an API key. |
| `--port` | integer | `8095` | The TCP port. `0` asks the OS for a free port; the server prints the port it got. |
| `--api-key` | string, comma separated | `$STRATA_API_KEY` or empty | Requires this key on the API: `Authorization: Bearer ...` or `x-api-key`. Empty means no key. The config's `api_key` is used when the flag is absent. |
| `--parallel` | `1..1024` | none (the config's `parallel`) | Requests this many engine slots from the engine without changing the saved config. Conflicts with `--batch` or `--slots` in the config's `args`. |
| `--replicas` | integer | none (the config's `replicas`) | Data-parallel replicas: split the `gpu` list into N groups and run one engine on each. See [MULTI_GPU.md](MULTI_GPU.md). |
| `--gpu` | `N`, or `0,2` for a layer split | none (the config's `gpu`) | The GPU to run on, numbered as `nvidia-smi` numbers it. A list is a layer split. |
| `--lazy` | flag | off | Start the API unloaded. The engine and the vision encoder start with the first request. See [DETAILS.md](DETAILS.md#start-the-api-without-loading-the-model). |
| `--idle-unload` | seconds (float) | none (the config's `idle_unload_s`, then never) | Unload the model after this many seconds without requests. The next request loads it again. |
| `--ready-file` | absolute path | none | Write a JSON instance record when the API is bound and the model state is known. See [Lifetime](#lifetime). |
| `--max-queue` | `0..4096` | none (the config's `max_queue`, then no limit) | Inference calls allowed to wait beyond the effective engine slots. `0` refuses overload instead of queueing. |
| `--queue-timeout` | `0.1..600` seconds | none (off) | The longest wait for the first engine turn. It needs `--max-queue` or `max_queue`. |
| `--min-free-vram-mib` | integer MiB | none (the config's `min_free_vram_mib`, then `0` = always load) | Load an unloaded model only when this much VRAM is free. Otherwise the server answers 503. |
| `--fit-max-tokens` | flag | on (kept for older scripts) | Clamp `max_tokens` to the remaining context instead of rejecting the request. |
| `--no-fit-max-tokens` | flag | off | Answer 400 when prompt plus `max_tokens` does not fit the context, like llama.cpp. A prompt that alone does not fit is always 400. |
| `--tokenizer` | directory | `<repo>/pack/full/tokenizer` | The pack tokenizer directory. The config's `tokenizer` key overrides the flag. A missing `vocab.json` stops the start with `--engine strata`. |
| `--api-monitor` | flag | off (the config's `api_monitor`) | Keep the last 100 requests' prompts and answers in memory for `/api-monitor`. |
| `--before-load` | command string | none (the config's `before_load`, a string or a list) | A command run before each load of the model, for example to free the GPU from another program. |
| `--mcp-config` | path | none | A JSON file with MCP servers in Claude Desktop's format. Its `mcpServers` block is added to the config's `mcp_servers`; a name in both takes the file's entry. |
| `--open` | flag | off | Open the local page in the browser when the model is ready. The config's `"open_browser": false` wins over the flag. |
| `--offline` | flag | off | No remote image URLs and no MCP or tool subprocesses. A config that names MCP servers, `before_load` or a network source is refused at start. |
| `--script` | string, repeatable | a short greeting | The mock engine's answer. Given more than once, requests get them in turn and the last one repeats. |
| `--script-cycle` | flag | off | With several `--script` values, repeat them by position instead of staying on the last one. |
| `--script-delay` | seconds (float) | `0.0` | Wait between the mock engine's tokens, so a client can cancel a live turn. |
| `--shutdown-timeout` | `0..300` seconds | none (off) | Opt-in graceful drain deadline. Owned requests are cancelled after it. The terminal passes `30`. |
| `--slot-save-path` | directory | none (the config's `slot_save_path`) | Enables `POST /slots/0?action=save\|restore {"filename": NAME}`. A relative path is taken from the working directory; the config's key is taken from the config's `cwd`. |
| `--owner-fd` | file descriptor | none | Hidden (`argparse.SUPPRESS`). The local runtime host passes an inherited pipe: end of file means the host exited. See [Lifetime](#lifetime). |
| `--batch-groups` | `auto` or a positive integer | none | Pipeline groups for the engine, without changing the saved config. It needs a config with `args` and conflicts with a saved `--batch-groups`. |
| `--batch-telemetry` / `--no-batch-telemetry` | flag pair | none (inherits the environment, then the config's `batch_telemetry`) | Opt in or out of low-rate engine decode-window samples for `/v1/telemetry`. |

`--batch-telemetry` sets `STRATA_BATCH_TELEMETRY=1` in the engine's environment and
`--no-batch-telemetry` removes the variable (`configure_batch_telemetry_env`,
`serve/server.py:1096`). Only an explicit setting changes an inherited value.

## Run config

The run config is the JSON file that setup writes as `strata-<model>.json`. The server reads a
UTF-8 file and accepts a byte order mark. A file that is not valid JSON stops the start and names
the line and column.

An engine option is kept in `args` as a list of separate strings. An element that joins a flag and
its value in one string (`"--ple-io mmap"` or `"--ple-io=mmap"`) is split at start
(`split_joined_args`, `serve/server.py:8003`).

### Model and engine

| Key | Type | Default | Description |
|---|---|---|---|
| `exe` | string | none | The engine binary. `component:strata-engine` means the engine package for this PC. |
| `args` | list of strings | none | The engine's own options. See [ENGINE_FLAGS.md](ENGINE_FLAGS.md). |
| `cwd` | string | the server's working directory | The engine's working directory. Relative paths in the config use it. |
| `env` | object | none | Extra environment for the engine process. |
| `lib_dirs` | list of strings | none | Folders put first on the engine's library search path. |
| `log` | path | none | The engine's log file. |
| `tokenizer` | directory | `pack/full/tokenizer` | The tokenizer the server uses to count prompt tokens. |
| `arch` | `qwen4exp` or `qwen35moe` | `qwen4exp` | The model family. It selects the API model name and the model card's sampling. |
| `model_name` | string | from `arch` | The model name the API reports and the aliases answer to. `qwen4exp` gives `qwen3.8-flash-next`, `qwen35moe` gives `qwen3.6-35b-a3b`. |
| `gpu` | integer or `"0,2"` | none | The card or cards. A list is a layer split. |
| `gpu_order` | `as_given` | none | Keeps the `gpu` order instead of reordering the cards. |
| `backend` | `hip` | none | The AMD path. It selects `HIP_VISIBLE_DEVICES` and the sysfs readings. |
| `hip_ordinal` | integer | none | The AMD card numbering base. |
| `layer_split` | `auto` or a layer number | `auto` | The engine's `--layer-split` value. See [MULTI_GPU.md](MULTI_GPU.md). |
| `split_skip_if_fits` | boolean | off | Opt-in. An auto split runs on the first card alone when that card holds every profiled expert and the KV cache. |
| `expert_profile_save` | path | none | Where the engine saves the expert-tier profile it learned. A relative path uses the config's `cwd`. |
| `expert_profile_save_every` | number of minutes | `10` (`0` = at `QUIT` only) | How often that profile is saved. |
| `conversation_cache_keep_root`, `conversation_cache_disk_keep_root` | boolean | none | `false` adds `--no-conversation-cache[-disk]-keep-root`. |
| `parallel` | integer | none | Requested engine slots. See [BATCHING.md](BATCHING.md). |
| `batch_telemetry` | boolean | none | The default of `--batch-telemetry`. |
| `batch_groups` | not read | - | Use `--batch-groups` or add the engine option to `args`. |
| `replicas` | integer or list of `{"gpus": [...]}` | none | Data-parallel replicas. |
| `replica_cpus` | boolean | `true` | Give each replica its own CPU set. |
| `replica_load_tokens` | integer | the replica loader's own value | The prompt length used to load each replica. |
| `vision` | object | none | The image encoder. See [IMAGES.md](IMAGES.md). |
| `images` | object | none | Image generation. See [IMAGES.md](IMAGES.md). |

The `vision` block has its own keys: `lazy`, `idle_s`, `vram_mib`, `elastic`, `elastic_mib`,
`elastic_idle_s`, `cuda_device`, `exe`, `mmproj`, `model`, `gpu`, `threads`, `max_tokens` and
`min_tokens`. The `images` block has its own too. Both are described in [IMAGES.md](IMAGES.md).

### Network and security

| Key | Type | Default | Description |
|---|---|---|---|
| `host` | string | `127.0.0.1` | The listen address. `--host` wins. |
| `api_key` | string or list of strings | empty | Keys accepted on the API. A comma in a string separates keys; a list gives a key that contains a comma. |
| `telemetry_read_key` | string | empty | A read-only key for `/v1/telemetry`. It must differ from every inference key. `$STRATA_TELEMETRY_KEY` wins. |
| `cors_origins` | string or list | none | Origins allowed on `/v1/*` by CORS. `*` allows any origin. See [SAFETY.md](SAFETY.md). |
| `trusted_origins` | string or list | none | Extra origins allowed to use the control endpoints. A wildcard is refused. |
| `allowed_hosts` | string or list | none | Extra `Host` names the server answers to, for a reverse proxy. `*` turns the Host check off. `$STRATA_ALLOWED_HOSTS` sets it too. |

The server does not read `port` from the run config. The terminal passes `--port`.

### Sampling defaults

The `sampling` block gives the values a request leaves out. A field in the request always wins, and
a field set to `null` takes the default. A bad value stops the start.

| Key | Type | Default | Description |
|---|---|---|---|
| `sampling.temperature` | number `>= 0` | the model card's (`1.0` with thinking, `0.7` without) | `0` is greedy. |
| `sampling.top_p` | `0 < x <= 1` | the model card's (`0.95` with thinking, `0.8` without) | Nucleus sampling. |
| `sampling.top_k` | integer `1..64` | the model card's `20` | The sampled path takes at most 64 candidates. |
| `sampling.min_p` | `0 <= x <= 1` | `0` (off) | Minimum probability. |
| `sampling.presence_penalty` | number `>= 0` | the model card's (`1.5` without thinking, `0` with) | Presence penalty. |
| `sampling.frequency_penalty` | number `>= 0` | `0` (off) | Frequency penalty. |
| `sampling.repetition_penalty` (also `repeat_penalty`) | number `> 0` | `1` (off) | Repetition penalty. |
| `sampling.penalty_last_n` (also `repeat_last_n`) | integer `>= 0` | the engine's `64` | The penalty window. |
| `sampling.seed` | positive integer | none | A fixed seed. |
| `sampling.strata_tune` | object | none | Per-request tuning for every request. Only `pcie_frac`, `spec_min_p` and `aux_cpus` are kept from this block. |
| `sampling.experimental_speed_projection` | boolean | on | `experimental`. The default of the request key of the same name, when the engine was started with the control vector. |

Two names for the same field in one block stop the start. Unknown keys are named at start and
ignored.

### Reasoning

| Key | Type | Default | Description |
|---|---|---|---|
| `reasoning_effort` | `none`, `low`, `medium` or `high` | the model's | The thinking level of requests that name none. It changes the answers. |
| `reasoning_budget_tokens` | integer `>= 0` | none | Cap the thinking of every request. `0` means no cap. It changes the answers. |
| `reasoning_loop_recovery` | `false`, `"stop"` or `"recover"` | `false` | Opt-in. Ends or recovers a reply that loops in its thinking. |
| `reasoning_close_retry` | boolean | `false` | Opt-in. Close the thinking once and continue when a reply ends inside it. |
| `effort_position` | `start` or `end` | `start` | Where a non-default effort goes in the prompt. `end` keeps the conversation cache when the effort changes. |
| `anthropic_thinking` | `model` or `on_request` | `model` | Whether an Anthropic request that does not ask for thinking still thinks. It changes the answers. |

### Cache and lifecycle

| Key | Type | Default | Description |
|---|---|---|---|
| `lazy_load` | boolean | `false` | Start unloaded; the first request loads the model. Text only. |
| `idle_unload_s` | number `>= 0` | `0` (never) | Unload after this many seconds without requests. |
| `min_free_vram_mib` | integer | `0` (always load) | Load only when this much VRAM is free. |
| `fit_max_tokens` | boolean | `true` | Clamp `max_tokens` to the remaining context. `false` answers 400 like llama.cpp. |
| `engine_silence_s` | number `>= 0` | `300` | End a request when the engine prints nothing for this long. `0` waits. |
| `max_queue` | integer `0..4096` | none | Pending inference calls beyond the engine slots. |
| `queue_timeout` | number `0.1..600` | none | The longest wait for the first engine turn. It needs `max_queue`. |
| `before_load` | string or list | none | The command run before each load. |
| `before_load_timeout_s` | number | `120` | The longest run time of that command. |
| `before_load_strict` | boolean | `false` | Refuse with 503 when the command fails. |
| `before_load_ignore_failure` | boolean | `false` | Load when the command fails, without a VRAM check. |
| `slot_save_path` | directory | none | Enables saving and restoring the conversation the engine holds. |
| `api_monitor` | boolean | `false` | Keep the last 100 requests in memory for `/api-monitor`. |
| `open_browser` | boolean | `true` | Open the page when the model is ready. `false` wins over `--open`. |
| `aliases` | list, or one comma-separated string | none | Other model names the server lists and answers to. |
| `conversation_cache_disk` | path | none | A disk tier for the conversation cache. |
| `conversation_cache_disk_gib` | number | none | Its size limit. |
| `vram_elastic` | boolean | `false` | Opt-in elastic VRAM. See [VRAM_ELASTIC.md](VRAM_ELASTIC.md). |
| `vram_segment_mib` | integer | none | The elastic segment size. |
| `answer_room` | boolean | `true` | `false` disables the room check before the answer. |
| `tool_call_continuation` | boolean | `false` | Opt-in continuation of a tool call the answer ended inside. |
| `tool_call_recovery` | boolean | `false` | Opt-in recovery of tool calls in nearby forms. |
| `repeat_stop_tokens` | integer `>= 0` | `256` | Stop a reply that repeats this many tokens. `0` turns it off. |
| `codex_thread_titles` | boolean | `false` | Opt-in. Answers the Codex title request. |
| `codex_compaction_cache` | boolean | `false` | Opt-in. Keeps the prompt tools of a Codex compaction. |
| `prevent_sleep` | boolean | `false` | Keep the PC awake while the server runs. Windows only. |
| `env_to_user` | boolean | `false` | Opt-in. Repeat the environment block to the user. |

### MCP

| Key | Type | Default | Description |
|---|---|---|---|
| `mcp_servers` (or `mcpServers`) | object | none | The MCP servers. Each server names `command`, `args`, `env` and `cwd` (stdio), or `url` and `headers` (HTTP). |
| `mcp.timeout_s` | number | `60` | A call's timeout. |
| `mcp.start_timeout_s` | number | `120` | A start's timeout. |
| `mcp.max_result_chars` | integer | `20000` | The cap on a tool result sent to the model. |
| `mcp.max_rounds` | integer | `8` | The tool rounds of one request. |

The tools reach the model as OpenAI function tools named `<server>__<tool>`. See
[MCP_SERVER.md](MCP_SERVER.md).

### Keys this list does not have

The server does not read these keys. Do not put them in the run config.

| Key | Where the value comes from |
|---|---|
| `port` | `--port`, or the terminal's saved profile. |
| `context` | The engine's `--max-context` option in `args`. The server reads the window from the engine. |
| `data_dir` | Setup and the terminal only. |
| `model_id`, `name` | `model_name` is the key. The ready file reports it as `model_id`. |

## Endpoints

`do_GET` (`serve/server.py:6583`) and `do_POST` (`serve/server.py:6927`) are the routing points. A
trailing slash does not change the path. `/v1/v1/messages` and `/v1/v1/messages/count_tokens` are
rewritten to their single-`v1` forms, for a client given a base URL that already ends in `/v1`.

When an API key is set, every GET path except `/`, `/web/*`, `/fonts/*`, `/health`, `/api-monitor`
and `/v1/images/files/*` requires it, and every POST path requires it. `/v1/telemetry` accepts the
inference key or the telemetry key. The control paths
(`/settings`, `/config`, `/unload`, `/load`, `/slots/*`, `/v1/load`, `/v1/unload`, `/v1/vram`) also
require `Content-Type: application/json` and an origin that is not foreign.

| Method | Path | Purpose | Request | Response |
|---|---|---|---|---|
| GET | `/` | The web page. | - | `text/html`. |
| GET | `/web/*` | The page's styles, script and icon sprite. Only `.css`, `.js` and `.svg` inside `serve/web`. | - | The file, `Cache-Control: no-cache`. |
| GET | `/fonts/*` | A `.woff2` font from `serve/web/fonts`. | - | The font, cached for one day. |
| GET | `/health`, `/api/health` | A liveness check with no key. | - | `{"status": "ok", "max_context", "model", "images", "api_key", "image_generation", "loaded", "service"}`. |
| GET | `/v1/models`, `/models` | The model list in OpenAI form. An unloaded model is listed only with `idle_unload_s` or an unloaded engine. Aliases are listed with `alias_of`. | - | `{"object": "list", "data": [...]}`. |
| GET | `/v1/status` | The server's own description: model, window, dialects, concurrency, activity, the last timings, VRAM, machine. | - | A JSON object. |
| GET | `/v1/telemetry` | The versioned telemetry snapshot: slots, groups, replicas, activity, topology, quality flags. | - | A JSON object. Key: inference or telemetry key. |
| GET | `/status` | The live status, including the end of the last answer. | - | A JSON object. Requires the API key. |
| GET | `/props` | llama.cpp's properties for a front end: the default sampling, `n_ctx`, `total_slots`, the template source, `is_sleeping`. | `?model=` | A JSON object. 404 for an unknown model; 503 when the engine is down. |
| GET | `/slots` | One entry per engine slot: `id`, `n_ctx`, `is_processing`, `n_prompt_tokens`. | - | A JSON list. Empty when the model is unloaded. |
| POST | `/slots/{id}` | Save or restore the engine's conversation. Only slot `0`. Needs `--slot-save-path`. | `?action=save\|restore`, `{"filename": NAME}` | `{"id_slot", "filename", "n_saved"\|"n_restored", "n_written"\|"n_read", "timings"}`. 501 when disabled or with `parallel`; 400 for a bad slot, action or name; 404 for a missing file. |
| GET | `/metrics` | What the Monitor tab shows: engine facts, the live state, the last 12 requests, hardware, one minute of history. | `?requests=all` | A JSON object. With a scraper's `Accept`, the Prometheus text body (see below). |
| GET | `/metrics/prometheus` | The Prometheus text format, vLLM's metric names (`serve/prometheus.py`). | - | `text/plain; version=0.0.4`. 503 until the first snapshot. |
| GET | `/api/requests` | The monitor's retained requests. Only with `api_monitor`. | `?id=` | `{"requests": [...], "retention": 100, "persistent": false, ...}`; 404 when the record is gone. |
| GET | `/api-monitor` | The monitor page. Only with `api_monitor`. | - | `text/html`. |
| GET | `/settings` | The Chat settings other apps share. | - | `{"shared": bool, "defaults": {...}}`. |
| POST | `/settings` | Set or clear those defaults. | `{"defaults": {...}}` or `{"defaults": null}` | The new `{"shared", "defaults"}`. Keys: `reasoning_effort`, `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`, `reasoning_budget_tokens`, `experimental_speed_projection`. |
| GET | `/config` | The editable keys of the run config, as `/config`'s Settings view shows them. 404 without `--config`. | - | `{"file", "keys": [{key, value, help, kind}], "note"}`. |
| POST | `/config` | Change only the keys it names. The earlier file is kept as `.bak`. The change is used at the next start. | `{"set": {KEY: VALUE}}` | The same view as `GET /config`, plus `"changed"`. 400 for a bad value. |
| POST | `/unload` | Stop the engine between requests. | `?wait=SECONDS` | `{"status": "unloaded"\|"not loaded"\|"busy"\|"unsupported"}`. 409 when busy, 503 when the engine cannot be ended. |
| POST | `/load` | Load the model now. | - | `{"status": "loaded"}`; 503 with `Retry-After` when the GPU is busy. |
| POST | `/v1/unload` | The versioned unload. | `{"model": ...}`, `?wait=` | `{"status", ...v1_status}`. |
| POST | `/v1/load` | The versioned load. | `{"model": ...}` | `{"status", ...v1_status}`; 409 when a request runs or waits. |
| POST | `/v1/vram` | Change the VRAM the engine leaves free, between requests. Opt-in: needs `--vram-elastic` or GPU-only KV growth. | `{"reserve_mib": N \| [N, ...] \| null}` | The engine's figures: `expert_slots`, `expert_cache_mib`, `vram_free_mib`, one value per GPU. |
| GET | `/mcp` | The MCP servers, their state and their tool count. | - | `{"servers": [...], "tools": N}`. |
| POST | `/v1/chat/completions` | OpenAI Chat Completions. | A JSON body; see below. | JSON, or SSE with `"stream": true`. See [DETAILS.md](DETAILS.md#using-it). |
| POST | `/v1/messages` | Anthropic Messages. | A JSON body; see below. | Anthropic JSON, or Anthropic SSE events. |
| POST | `/v1/messages/count_tokens` | The prompt's token count. The model does not run. | A Messages body | `{"input_tokens": N}`. |
| POST | `/v1/responses` | OpenAI Responses, stateless. | A JSON body; see below. | Responses JSON, or typed SSE events. |
| GET, POST | `/v1/responses/{id}` | Retrieve, delete, cancel or compact. This server stores nothing. | - | `POST`: 404 with `code: "not_found"`. `GET`: the generic 404. `DELETE`, `PUT` and `PATCH` are not routed: 501. |
| POST | `/v1/images/generations` | One image generation job. Needs the `images` block. | `prompt`, `size`, `n`, `steps`, `seed`, `response_format` (`b64_json` or `url`), `output_format` (`png`) | An OpenAI image response. 404 when image generation is not set up. See [IMAGES.md](IMAGES.md). |
| GET | `/v1/images/files/{name}` | The PNG of a `url` response. The 32-hex-digit name is the access. | - | `image/png`, cached for one day; 404 when the file is gone. |
| OPTIONS | any | A CORS preflight. It carries no credentials and needs no key. | `Origin` | 204 with the CORS headers, only for an allowed origin. |
| any | any other path | - | - | 404 `{"error": {"message": "not found"}}`. |

`/v1/tune` does not exist. Per-request tuning is the `strata_tune` object in the request body; see
[Per-request tuning](#per-request-tuning).

### Chat Completions and Messages fields

These fields are read in the code. A field that is not listed here has no effect on the answer.

| Dialect | Fields | Source |
|---|---|---|
| Chat Completions | `model`, `messages`, `tools`, `tool_choice`, `stream`, `stop`, `max_tokens`, `max_completion_tokens`, `response_format`, `logit_bias`, `strata_mcp`, `chat_template_kwargs` | `serve/server.py:7353`, `serve/frontend.py:533` |
| Both dialects | `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` (also `repeat_penalty`), `penalty_last_n` (also `repeat_last_n`), `seed` | `check_request_sampling`, `serve/server.py:8219` |
| Chat Completions reasoning | `reasoning_effort`, `reasoning: {"effort": ...}`, `chat_template_kwargs.enable_thinking`, `chat_template_kwargs.reasoning_effort`, `chat_template_kwargs.preserve_thinking` | `serve/frontend.py:558-571` |
| Chat Completions messages | `role` (`developer` becomes `system`), `content` (text and image parts), `reasoning_content`, `tool_calls` | `serve/frontend.py:533-556` |
| Messages | `model`, `system`, `messages`, `tools`, `tool_choice`, `max_tokens`, `stop_sequences`, `stream`, `thinking`, `output_config.effort`, `reasoning_budget_tokens` | `serve/frontend.py:604`, `serve/server.py:7607` |
| Responses | `input`, `instructions`, `tools`, `tool_choice`, `text`, `reasoning`, `temperature`, `top_p`, `max_output_tokens`, `include`, `parallel_tool_calls`, `client_metadata`, `metadata`, `stream` | `serve/responses.py` |

`strata_prefix` is read for every dialect (`serve/server.py:4902`).

`n` and `stream_options` are not read. The server returns one choice, and `logprobs` is always
empty. `response_format` takes `json_object` and `json_schema`; a schema is checked after
generation, not enforced during it (`structured_output.constrained_decoding` is `false` in
`/v1/status`). See [DETAILS.md](DETAILS.md#json-response-formats).

The Responses API refuses `previous_response_id`, `conversation`, `background` and `prompt` with
`400` and `unsupported_parameter`, because nothing is stored (`serve/responses.py:437-448`). Hosted
tools are left out of the prompt.

`stop` takes at most 4 strings. `stream` is per request; there is no server-wide setting.

## Per-request tuning

`strata_tune` is an object in the request body. The server turns it into engine keys on the `GEN`
line of that request only. It is not stored.

| Key | Type | Range | Default | Effect |
|---|---|---|---|---|
| `pcie_frac` | number | `0.0..1.0` | the engine's `--pcie-frac`, else the model's default | The share of the expert misses read over PCIe instead of from the CPU pool. It changes the speed. |
| `spec_min_p` | number | `0.0..1.0` | the engine's `--spec-min-p`, else `0` | How sure the draft layer must be to extend a verify window. It changes the answers: `placement`. |
| `prefill_pipe` | number | `0..262144` | the engine's `--prefill-pipe`, else off | The prompt chunk size of a layer split, in tokens. `0` turns it off, `1` only evens the buffers out. It changes the speed. |
| `aux_cpus` | boolean or `0`/`1` | `0` or `1` | the engine's `--aux-cpus` | `1` moves the engine's spare threads to the spare CPUs, `0` leaves them where they were created. It changes the speed, never a result. |

`prefill_pipe` is accepted in a request. The config's `sampling.strata_tune` keeps only
`pcie_frac`, `spec_min_p` and `aux_cpus` (`sampling_defaults_from_config`, `serve/server.py:8161`).

A value of the wrong JSON type, or outside its range, is not sent. The engine ignores an unknown key.

### Precedence

One rule holds for these keys: the request wins over the start-up flag, and the flag wins over the
environment variable.

1. The request's `strata_tune` value. The engine takes it for that request only
   (`src/program/generate.cpp:13674-13710`).
2. The engine's flag: `--pcie-frac`, `--spec-min-p`, `--prefill-pipe`, `--aux-cpus`. It is the
   start-up value, and it is what a request without the key uses.
3. The environment variable: `STRATA_AUX_CPUS` sets `--aux-cpus` when the flag is absent
   (`src/program/generate.cpp:3036`), and `STRATA_PREFILL_PIPE` sets `--prefill-pipe`
   (`src/prefill/prefill.cpp:2608-2614`).

The engine states the `aux_cpus` case: "a request's strata_tune aux_cpus=0|1 beats it"
(`src/program/generate.cpp:1097`). A request's `pcie_frac` also stops the automatic refinement of
the share for that request (`req_pcie_set`).

A setting that changes the numbers of the answers is marked in
[ENGINE_FLAGS.md](ENGINE_FLAGS.md) and [TUNING.md](TUNING.md). Do not put a quality setting in
`strata_tune` without a measurement: the same engine and the same prompt must give the answer you
accept.

## Lifetime

### Ready file

`--ready-file` must be an absolute path, its parent must exist, and the file must not exist
(`serve/ready.py`). The server writes one JSON object with mode `0600` when the API is bound and
the model state is known:

| Field | Meaning |
|---|---|
| `schema_version` | `1`. |
| `instance_id` | The instance's identity. |
| `pid` | The server's process ID. |
| `api_url`, `bound_host`, `bound_port` | The address a client can use. `--port 0` reports the port the OS gave. |
| `model_id` | The model name. |
| `model_state` | `ready`, or `unloaded` with `--lazy`. |
| `api_state` | `ready`. |
| `created_at` | Unix time. |

The server removes the file at exit only when its `instance_id` and `pid` match. A killed process
leaves the file behind; a consumer must compare the instance ID with live telemetry before it acts.
A write failure stops the server with exit code 6.

### Lazy load, idle unload and VRAM

- `--lazy` (or `"lazy_load": true`) starts the API without the engine. The first request loads the
  model. The vision encoder also starts with the first request.
- `--idle-unload` (or `idle_unload_s`) checks for idle time every `min(30, idle_unload_s / 4)`
  seconds, never less than one second, and never unloads while a request runs or waits. The next
  request loads the model again.
- `--min-free-vram-mib` is checked before a load. A card with less free VRAM gives 503, and the
  request does not run.
- `before_load` runs before each load, including a load after an idle unload. Its failure is
  handled by `before_load_strict` and `before_load_ignore_failure`.

### Queue and admissions

- With `parallel`, the engine has N slots. `max_queue` adds pending calls beyond them.
- `max_queue` of `0` refuses overload: the admission limit is the slot count, and the next request
  gets `429` with `code: "admission_full"` and `Retry-After: 1`.
- `queue_timeout` bounds the wait for the first engine turn. A wait that expires gives `429` with
  `code: "queue_wait_timeout"`.
- A client that leaves while its request waits cancels it. The telemetry counts these as
  `client_disconnect_cancels`.

### Owner guard and shutdown

- `--owner-fd` is for the local runtime host. On POSIX the server gets its own process group and
  reads an inherited pipe; on Windows the pipe is its standard input. End of file means the host
  exited and becomes the Ctrl+C path. The guard grace is `shutdown_timeout + 40` seconds.
- SIGTERM (docker stop, systemd), SIGHUP (the terminal closed, ssh dropped) and the Windows console
  events take the Ctrl+C path. SIGINT keeps Python's own handler: a second Ctrl+C ends the engine
  at once.
- Without `--shutdown-timeout`, the stop cancels the work at once. With it, the server stops
  admission, waits up to that deadline for owned requests, then cancels them.
- The stop order is the HTTP server, the warm-up thread, the engine (`QUIT`, then the wait
  timeouts), the vision encoder and the MCP hub. The engine's waits are 20 s for `QUIT`, 5 s for
  TERM and 20 s for KILL (`serve/server.py:100`).

### Exit codes

| Code | Meaning | Source |
|---|---|---|
| `0` | A normal stop. | `serve/server.py:8885` |
| `2` | A refused start: an empty API key, a bad key, a non-loopback bind without a key, a bad telemetry key, or the same key for both. | `serve/server.py:8472-8480`, `8690` |
| `5` | The engine component for this PC is missing. | `serve/server.py:8301` |
| `6` | The ready file could not be written. | `serve/server.py:8856` |
| `8` | The drain deadline expired and requests were cancelled. | `serve/server.py:8885` |

An argparse error also exits with code 2, as Python does.

### Engine child process contract

The server starts the engine once as a child process (one child per replica group) and keeps it. It
speaks line commands on the engine's standard input and reads line answers from its standard output:

| Direction | Line | Meaning |
|---|---|---|
| server to engine | `GEN <max_new><keys> <ids...>` | Generate, with the sampling and tuning keys between `max_new` and the ids. |
| server to engine | `BGEN <slot> <max_new><keys> <ids...>` | The same, on a batch slot. |
| server to engine | `GENI`, `BGENI` | The same for a request with images: the embedding file path follows the ids. |
| server to engine | `STOP`, `BSTOP <slot>` | Stop the request. |
| server to engine | `BYIELD <slot>` | Park the request. |
| server to engine | `VRAM <reserve_mib>` | Change the VRAM the engine leaves free. |
| server to engine | `SAVE <path>`, `RESTORE <path>` | The conversation file of a slot. |
| server to engine | `QUIT` | A clean stop. |
| engine to server | `READY`, `PP`, `T`, `DONE`, `ERR`, `BSTAT`, `SERR`, `SESSION`, `SWAIT` | The start, prompt progress, tokens, the end of a request, an error, and the telemetry samples. |

An engine that dies is started again by the next request, with the same command line (issue #27).
An engine that prints nothing for `engine_silence_s` is ended. An engine that prints nothing, uses
no CPU time and moves no bytes for `STRATA_ENGINE_STALL_S` (default 90) is frozen: it is ended and
restarted on the next request.

## HTTP status codes

| Code | When |
|---|---|
| 200 | The answer, or the control action, succeeded. |
| 204 | A CORS preflight. |
| 400 | A bad request body, a bad value, a malformed request, or a prompt that alone does not fit the context. |
| 401 | A missing or wrong API key. |
| 403 | A control endpoint from a foreign origin. |
| 404 | An unknown path, model or file, or a stored response that does not exist. |
| 409 | The model is busy and the action needs it idle. |
| 413 | A body larger than the limit: 256 MiB, or 64 KiB on a control path. `STRATA_MAX_BODY_MIB` raises the first. |
| 415 | A control endpoint without `Content-Type: application/json`. |
| 429 | The admission limit is full, or the queue wait expired. |
| 500 | The server or the image job failed. |
| 501 | A feature that is not enabled: slot save, or slot save with `parallel`. |
| 502 | A structured output check failed. |
| 503 | The model is unloaded and not loadable now, the GPU is busy, the engine is starting, the engine died, or the instance is draining. |

A `503` from a load or a busy GPU carries `Retry-After` when the server knows a delay.

## Environment variables of the server and the terminal

These `STRATA_*` names are read by the Python code of the server, the terminal or the control
package. The engine's own variables are in [ENVIRONMENT.md](ENVIRONMENT.md); a name that appears in
both lists is an engine setting.

### Server

| Name | Default | Description |
|---|---|---|
| `STRATA_API_KEY` | empty | The API key when `--api-key` is absent. A comma separates several keys. |
| `STRATA_ALLOWED_HOSTS` | empty | Extra `Host` names, as the config's `allowed_hosts`. |
| `STRATA_TELEMETRY_KEY` | empty | The read key of `/v1/telemetry`. It must differ from every inference key. |
| `STRATA_OFFLINE` | empty | Set to `1` for the engine by `--offline`. |
| `STRATA_MAX_BODY_MIB` | `0` (256 MiB) | The request body limit in MiB. |
| `STRATA_HTTP_BACKLOG` | `256` | The listen backlog, minimum 5. |
| `STRATA_ENGINE_READY_S` | `900` | The longest start time before `READY`. `0` waits for ever. |
| `STRATA_VISION_READY_S` | `300` | The same for the vision encoder. |
| `STRATA_VISION_ENCODE_S` | `300` | The same for one image encode. |
| `STRATA_ENGINE_STALL_S` | `90` | The longest time an engine may show no work before it is ended as frozen. `0` turns it off. Needs psutil. |
| `STRATA_START_RETRIES` | `3` | The retries of a start that died before `READY`. |
| `STRATA_ADMIT_SHORT_FIRST` | `0` (off) | A prompt of at most this many tokens goes before longer ones. |
| `STRATA_ADMIT_MAX_WAIT_S` | `30` | A longer prompt that waited this long counts as short. `0` turns it off. |
| `STRATA_PARALLEL_SOLO` | on | A request left alone in its batch slot goes back to the solo path with its drafts. `0` keeps it in the slot. |
| `STRATA_REPLICA_START` | empty | `serial` starts the replicas one after another. |
| `STRATA_DEBUG` | empty | `1` writes request diagnostics and the raw model text to the log. |
| `STRATA_BATCH_TRACE` | empty | `1` traces the engine's line exchange of a batch request on standard error. |
| `STRATA_REQUEST_LINES` | empty | `1` writes one stdout line per finished request, from the engine's log. |
| `STRATA_KEEP_LEFTOVERS` | empty | `1` skips the end of engines an earlier run left behind. |
| `STRATA_KEEP_EMPTY_TURNS` | empty | `1` keeps empty assistant turns in the prompt, as before #843. |
| `STRATA_NO_PRESET` | empty | Set by a tuning worker: do not apply a saved tuning on top. |
| `STRATA_OWNER_ATTEST_KEY` | empty | Set by the runtime host: the private launch attestation key. |
| `STRATA_LIFELINE` | empty | `watcher` forces the portable watcher path. |
| `STRATA_RUNTIME_PROFILE` | empty | The profile name the process records belong to. |
| `STRATA_SYCL_BIN` | `build-sycl-aot/strata` | The SYCL wrapper's binary, when the config's `env` does not name one. |
| `STRATA_DROP_CACHE_ON_EXIT` | `0` | `1` releases the model files' cache after a second Ctrl+C. |

### Terminal and control

| Name | Default | Description |
|---|---|---|
| `STRATA_URL` | `http://127.0.0.1:8095` | The server a client command talks to. |
| `STRATA_CONNECTION` | empty | The saved connection to use. |
| `STRATA_CONNECTION_DIR` | the per-user folder | Where the saved connections are kept. It must be absolute. |
| `STRATA_INSTALL_DIR` | the repository root | Where `setup.py` and the install tree are. |
| `STRATA_COMPONENTS_DIR` | the per-user component folder | Where the engine packages are. |
| `STRATA_ENGINE_BACKEND` | detected | The engine family to use instead of the detected one. |
| `STRATA_REQUIRE_SIGNED_COMPONENTS` | empty | `1` refuses an unsigned engine package. |
| `STRATA_PROFILE_DIR` | `~/.config/strata/profiles` on Linux, `%APPDATA%\Strata\profiles` on Windows | The saved serving profiles. |
| `STRATA_PRESET_DIR` | beside the profiles | The saved tunings. |
| `STRATA_TUNE_DIR` | the state folder's `tuning` | The tuning jobs' records. |
| `STRATA_TUNE_GPU_LOCK` | empty (no lock) | The lock file that holds a GPU for one tuning job. |
| `STRATA_TIMESKIP_DIR` | the state folder's `time-skip` | The Time-skip runs' records. |
| `STRATA_TIMESKIP_PROMPT` | set by the terminal | The prompt of the Time-skip run in progress. |
| `STRATA_TIMESKIP_ATTEMPT` | set by the terminal | The attempt number of that run. |
| `STRATA_RUNTIME_DIR` | the state folder's `runtime` | The runtime's own files. It must be absolute. |
| `STRATA_STATE_DIR` | empty | One more folder that counts as the user's own for the private-file repair. |
| `STRATA_HOME` | empty | One more folder that counts as the user's own data. |
| `STRATA_SETTINGS_FILE` | empty | The desktop app's settings file. Its presence marks the app's own processes. |
| `STRATA_DOCUMENTS_DIR` | the Documents folders | One more folder to look in for the desktop app's data folder. |
| `STRATA_GLYPHS` | detected | `ascii` or `unicode` forces the console glyph set. |
| `STRATA_CONTROL_INSTALL` | empty | `1` runs `setup.py` in the app's own installer mode. |
| `STRATA_NO_ACQUIRE` | empty | Set: no download is allowed, and an update is refused with exit code 5. |
| `STRATA_UPDATE_FEED` | the feed used last, else the project's | Where the signed update feed is. |
| `STRATA_UPDATE_ALLOW_HTTP` | empty | Set: the feed can use plain HTTP. |
| `STRATA_UPDATE_TRUSTED_KEYS` | the built-in keys | The update signing keys to trust. |
| `STRATA_APP_UPDATE_PUBKEY` | the built-in keys | The app update key to trust. |
| `STRATA_APP_VERSION` | none | The version the app update reports. |

The test suite also reads `STRATA_FAIL_ONCE` and `STRATA_TOKENIZER`. They are not settings.

The state folder is `%LOCALAPPDATA%\Strata` on Windows and `$XDG_STATE_HOME/strata`
(`~/.local/state/strata` by default) elsewhere (`strata_control/paths.py:13`).
