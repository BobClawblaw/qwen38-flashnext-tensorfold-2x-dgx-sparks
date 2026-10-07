# Changelog

This fork's own changes, newest first. The base recipe's history is in its repository
(sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark); its evidence is kept under `evidence/s1-*` to `s9-*`.

## 2026-10-07

- **Serial profile measured on its own boot** (`evidence/s10-tf066/serial-vs-concurrent/frozen-c1-serial-boot.out`): frozen prose 73.3 tok/s, structured 243.6 at one user, against 65.6 / 232.9 on the concurrent profile.
- **Quality gates rerun on 0.6.6** (`evidence/s10-tf066/quality`): GSM8K 95.6%, IFEval 85.0%, tools 60/60, repeated 4-grams 0.04%, `reasoning_effort` 5/5, needles 24/24 and 6/6 at 250k, JSON schema 30/30 on the serial profile. Patch part 3's attribute reads made defensive for engines built without `__init__` (upstream's tests do that); same runtime behaviour.
- **Upstream:** the copy drafts are offered as TensorFold pull request #468 with the full receipt. `docs/two-rank-vision-scope.md` scopes image input on two ranks (not started).
- **Copy drafts for Flash Next on CUDA** (`docker/patches/flashnext-tools-0.6.6.patch`, part 3). A reply whose last 8 tokens repeat earlier text drafts that text's continuation whole, up to the depth of 15. Quoting a passage 130 -> 345 tok/s, fixing its typos 124 -> 323, a rename refactor 135 -> 288; fresh prose and code unchanged; every reply byte-identical to one-token decoding. `TENSORFOLD_COPY_DRAFTS=0` turns it off. Offered upstream as a TensorFold pull request.
- **Concurrent profile: 16 streams** (`PROFILE=concurrent` sets `PARALLEL=16`). Measured 1 to 16 users, all replies equal to their solo runs: 714 tok/s aggregate on code and 653 on chat at 16, 1 to 8 users unchanged against 8 slots. TensorFold has no stream cap; the memory gate admits streams while their caches fit.
- **Tool calls 60/60.** `--tool-system TEXT` on the CUDA server (patch, part 1): a system instruction added to chat requests that offer tools and carry no system message of their own. The recipe's default `TOOL_SYSTEM` takes the sibling harness's 60-call set from 56 to 60; a client's own system prompt wins, text requests are untouched. The two misses were a declared optional argument the MLX 4-bit weights fill unasked; the chat template is byte-identical to vLLM's, and no engine setting reaches it.
- **`MAX_TOKENS` default 32768** (was 4096; the engine clamps it to the room left in the window). A thinking reply could spend 4096 inside its think block and return nothing.
- **Measured on 0.6.6** (`evidence/s10-tf066`): the frozen ruler, 512 and 1024-token replies at 1 to 4 users, one user on the serial against the concurrent profile (distinct prompts 79 against 55-62 tok/s), four streams at 250k tokens each, and the copy cells. Round cost is 29 ms at 2 drafts a round and 50-60 ms at 14; tokens per round is what the text allows.
- **Documented engine behaviours:** `--decode-share` defaults to 0 on CUDA, so a filling long prompt starves live replies; the concurrent decoder keeps 8 prompt states; GB10 ignores `nvidia-smi -lgc`.

## 2026-10-06

- **TensorFold 0.6.6** (`cb2ebf0`) replaces 0.6.2 and the PR141 port: one image, native two-rank `--parallel`, the 5-17% faster chain kernel, `TENSORFOLD_PREFILL_ROWS`, API keys, the Anthropic Messages API, live speeds on `/health`, `--name-priority`.
- **Declared-parameters rule** (patch, part 2): a Qwen XML tool call keeps only the parameters the offered tool declares, on the end parser, the CUDA reply parser and the streamer, so streamed and non-streamed arguments agree. The earlier overlay's `tools-patched.py` was byte-identical to upstream and was never in the image.
- **Full recipe tree** instead of an overlay: `run.sh` reads `.env.cluster` with environment precedence (`ENV_CLUSTER=/dev/null` for tests and CI); the keeper loop is gone; the systemd unit and wrapper are generic; tests and CI updated; `tools/bench_longctx_concurrent.py` added.
- **Measured on the first 0.6.6 boot:** prose 65.6, structured 232.9, count 1-400 at 220.8, 8 users 319 / 439 tok/s, tool calls 56/60 before the system instruction.
