# Qwen3.8-Flash-Next · TensorFold 0.6.6 · 2× DGX Spark

Serve [TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP) across two NVIDIA DGX Spark (GB10, 128 GB) nodes at tensor-parallel 2 with the [TensorFold](https://github.com/ashhart/TensorFold) engine: sixteen concurrent streams, the full 262,144-token window on both ranks, MTP speculative decoding (15 drafts at confidence 0.70), int8 KV cache, an OpenAI-compatible API, and a systemd unit that brings the pair up at boot.

This is a fork of [sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark](https://github.com/sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark) (MIT), whose guards, harness and evidence it keeps. What this fork changes:

- **TensorFold 0.6.6** (`cb2ebf0`) instead of 0.6.2. Upstream 0.6.4 merged two-rank `--parallel`, so the base recipe's `pr141-on-0.6.2.patch` and its second image are gone: one image, `PROFILE=concurrent` (`PARALLEL=16`) for sixteen streams. 0.6.3 to 0.6.6 also bring a 5-17% faster Flash Next chain kernel, `TENSORFOLD_PREFILL_ROWS`, API keys (`--api-key`), the Anthropic Messages API, `/health` with live decode and prefill speed, and `--name-priority`.
- **One patch, baked into the image:** [`docker/patches/flashnext-tools-0.6.6.patch`](docker/patches/flashnext-tools-0.6.6.patch). Two things. `--tool-system TEXT` on the CUDA server: a system instruction added to a chat request that offers tools and carries no system message of its own (a client's own system prompt wins; text requests are untouched). The recipe's default text takes the 60-call tool harness from 56 to 60 (Quality below). Copy drafts: a reply whose last 8 tokens repeat earlier text drafts that text's continuation whole, up to the depth, instead of an MTP chain, so quoting, editing and refactoring decode faster with byte-identical output (`TENSORFOLD_COPY_DRAFTS=0` turns it off). And a Qwen XML tool call keeps only the parameters the offered tool declares, on the end parser, the CUDA reply parser and the streamer. Provenance and scope: [`docker/patches/README.md`](docker/patches/README.md).
- **`.env.cluster`** next to `run.sh` holds the cluster's addresses and knobs (environment wins over the file). The shipped defaults in `recipe.yaml` stay generic.
- **systemd:** [`qwen38-tensorfold.service`](qwen38-tensorfold.service) + [`start-systemd.sh`](start-systemd.sh), `Type=oneshot` with `RemainAfterExit`.
- **int8 KV cache and one RoCE HCA** (`KV_DTYPE=int8`, `HCA=rocep1s0f1`) in this cluster's `.env.cluster`. With int8, sixteen streams each have room to grow to the full window (TensorFold reports 57 GiB free for their caches, 2.74 GiB for one at 262,144). Both-HCA NCCL (`rocep1s0f1,roceP2p1s0f1`, the base default) failed rank 1's boot once here (`ibv_query_port` errno 93); one HCA has been green since, and the base recipe measured one HCA within noise.

What changed when is in [`CHANGELOG.md`](CHANGELOG.md).

Qwen3.8-Flash-Next is a ~180B MoE (512 experts, top-10) with Gated DeltaNet, sparse attention, hyper-connections, hashed n-gram (PLE) tables and one MTP layer. The checkpoint is MLX affine 4-bit (group 32) on every linear, including the n-gram tables and the MTP head. It is the only Flash Next format TensorFold serves on two ranks: NVFP4 and EXL3 exports run on one GPU only. Pinned snapshot: `2b170fa6309d5d1ee380b35636075fac7945f286`.

TensorFold runs inside `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned), built locally from [`docker/Dockerfile`](docker/Dockerfile). Its kernels JIT-compile for sm_121 on the first start of each engine commit and are cached on the host after that. Rank 1 runs on the worker, rank 0 serves HTTP on the head; partials are all-gathered over NCCL on the QSFP RoCE link.

## Measured on this pair (TensorFold 0.6.6, concurrent profile, int8 KV)

`python3 bench_decode.py`, byte-identical to the vLLM sibling's frozen ruler (sha256 `6a9c64bd…`). Receipts: [`evidence/s10-tf066/`](evidence/s10-tf066/). The base recipe's tables (0.6.2, bf16 KV, one stream at a time) are in its README and under `evidence/s1-…s9-…`; its numbers are not repeated here.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
Conditions: streamed greedy, thinking off, max_tokens 200 (prose ends at EOS near 100), 3-run median; TensorFold 0.6.6 at TP=2 with PARALLEL=8, context 262144, kv int8, MTP up to 15 drafts at confidence 0.70, one RoCE HCA; the second bench_decode run on the boot (the first, right after a warm-up request, is bench-frozen-1.out).

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 65.6 | 65.6 | 0.06 s |
| prose (note 1) | 2 | 60.3 | 120.6 | 0.06 s |
| structured | 1 | 232.9 | 232.8 | 0.07 s |
| structured (note 2) | 2 | 210.7 | 417.5 | 0.09 s |

1. With PARALLEL=8 both streams decode together in shared lane rounds, so the aggregate is real concurrent throughput and the second stream's TTFT is its own prefill, not a queue wait.
2. Same as note 1.
<!-- END generated measured -->

### Longer and concurrent cells, same boot

| Cell | Result | Receipt |
|---|---|---|
| Count 1 to 400 (1,492 reply tokens), greedy, one stream | 220.8 tok/s, 110 verify rounds (13.6 tokens per round), 1,381 of 1,537 drafts accepted, TTFT 0.07 s | [`count-400-done-line.txt`](evidence/s10-tf066/count-400-done-line.txt) |
| `tools/vendor/bench_concurrent.py`, code prompt, greedy, 256 tokens, 1 user | 103.7 tok/s | [`bench-concurrent.out`](evidence/s10-tf066/bench-concurrent.out) |
| same, 8 users | 319.2 tok/s aggregate (40.0 per stream), slowest first token 0.17 s | same |
| chat prompt, greedy, 1 user | 87.9 tok/s | same |
| chat prompt, greedy, 8 users | 439.1 tok/s aggregate (55.1 per stream), slowest first token 0.17 s | same |

Every concurrent reply had the same token sha as the same request sent alone (`--alone`: 8 of 8 equal in both 8-user cells). A round's cost grows with the rows it verifies: about 29 ms when prose accepts 2 drafts a round, about 50-60 ms when counting, copying or code accept 6-14. Measured on one stream, 900-token replies: fresh prose 72 tok/s at 2.1 tokens a round; verbatim copy of a prompt passage 130 tok/s at 6.6; fixing typos in it 124 at 6.2; fresh code 127 at 5.7; renaming a variable across a file 135 at 6.4 (`evidence/s10-tf066/copy-vs-fresh.txt`). Against the base recipe's 0.6.2 numbers on the same ruler: prose c=1 65.6 against 69.6 (serial profile with CUDA graphs) and 67.0 (its PR141 port, measured here before the upgrade); structured c=1 232.9 against 249.5 and 223. The startup line's "0 decode graphs captured" counts only the serial engine's warm-up; with `PARALLEL=8` a lone stream gets its own graph slot and captures lazily on first use (`multi_solo.py`), and shared rounds with several streams run eager. The single-stream gap to the serial profile is 5-10% on repeated prompts and 27-43% on distinct ones (the per-prompt warm-up of the concurrent decoder; see the next section); `PARALLEL=1` removes it for a single user.

### Copy drafts: quoting, editing and refactoring

The recipe's patch adds copy drafts to Flash Next on CUDA: when a reply's last 8 tokens repeat text seen earlier (in the prompt or the reply), the round drafts that text's continuation whole, up to the depth of 15, instead of asking the MTP head. The verify step is untouched, so every reply below has the same token sha as the same request decoded one token a round (`"draft": false`). One stream, greedy, 900-token replies, this boot ([`copy-drafts/`](evidence/s10-tf066/copy-drafts/)):

| Cell | before (MTP only) | with copy drafts | tokens per round | one token a round |
|---|---:|---:|---:|---:|
| copy a passage verbatim | 130 | **345** | 15.5 | 48.5 |
| fix typos in a passage | 124 | **323** | 15.5 | 49.0 |
| rename a variable across a file | 135 | **288** | 13.9 | 48.5 |
| fresh prose (essay) | 72 | 71 | 2.1 | 48.7 |
| fresh code | 127 | 130 | 5.7 | 48.7 |

Fresh text has no earlier copy to draft from and is unchanged, as are the frozen ruler (prose 67.9, structured 235.7) and the 1 and 4-user cells (code 106 / 323, chat 87 / 274, every reply equal to its solo run and to one-token decoding). A plain round with no drafts costs about 20 ms; a copied chain of 15 verifies in about 45 ms. `TENSORFOLD_COPY_DRAFTS=0` in `EXTRA_ENV` turns it off.

### One user: serial against concurrent profile

Same image, same boot. `PARALLEL=1` (the serial engine, 48 CUDA graphs) against `PARALLEL=8` (the concurrent decoder). Receipts: [`evidence/s10-tf066/serial-vs-concurrent/`](evidence/s10-tf066/serial-vs-concurrent/).

| Cell, one user | `PARALLEL=8` | `PARALLEL=1` |
|---|---:|---:|
| 12 different prompts, 256 tokens (`tools/bench_distinct.py`) | 55 to 62 | **79** |
| frozen prose | 56 first pass, 66 repeated | **72** |
| frozen structured | 199 first pass, 240 repeated | **242** |
| code, 512 tokens | 87 | **111** |
| chat, 512 tokens | 78 | **97** |
| count 1 to 400 | 166 | **226** |

Rerun with the serial profile booted as the default (`PROFILE=serial`, fresh start, `frozen-c1-serial-boot.out`): frozen prose 73.3 tok/s, structured 243.6, TTFT 53-61 ms after the first request, against 65.6 and 232.9 on the concurrent profile's second pass.

The concurrent decoder pays a per-prompt warm-up, so repeated prompts look close to the serial engine and real traffic (a new prompt every time) does not: 27-43% slower. If the pair serves one client at a time, set `PARALLEL=1` in `.env.cluster` (it also serves `response_format`). Keep `PROFILE=concurrent` when several clients share it: 260-330 tok/s aggregate at four users, 650-715 at sixteen.

### Replies of 512 and 1024 tokens, 1 to 4 users

The cells an agent workload actually looks like. `tools/vendor/bench_concurrent.py` (TensorFold's), short prompts, greedy, 2 reps, every concurrent reply checked byte-equal to the same request alone (all equal). Aggregate tok/s, per-stream in brackets. Receipts: [`evidence/s10-tf066/replies-512-1024/`](evidence/s10-tf066/replies-512-1024/).

| Reply tokens | Workload | 1 user | 2 users | 4 users |
|---|---|---:|---:|---:|
| 512 | code | 106 | 197 (98) | 332 (83) |
| 512 | chat | 90 | 161 (81) | 278 (70) |
| 1024 | code | 86 | 155 (78) | 260 (65) |
| 1024 | chat | 99 | 180 (90) | 301 (75) |

At `PARALLEL=16` (now the concurrent profile's default), same ruler, 512-token replies, 2 reps, every reply byte-equal to its solo run ([`parallel-16/`](evidence/s10-tf066/parallel-16/)):

| Users | code, aggregate (per stream) | chat, aggregate (per stream) | slowest first token |
|---:|---:|---:|---:|
| 1 | 107 | 89 | 0.06 s |
| 4 | 331 (83) | 280 (70) | 0.08 s |
| 8 | 523 (66) | 459 (58) | 0.13 s |
| 16 | 714 (45) | 653 (41) | 0.24 s |

Memory after the 16-user cells: 35 GiB available on the head, 38 GiB on the worker. The engine has no stream cap; `--parallel` is a number and the memory gate admits streams while their caches fit (2.74 GiB each at the full window with int8 KV). The 8-user numbers vary by boot: 319 / 439 on one restart, 523 / 459 on this one, the base recipe's 520 / 425 on 0.6.2.

The same with a 1,000-token system prompt in front (`tools/bench_longctx_concurrent.py --prompt-tokens 1000 --tokens 1024`, distinct prompts per stream, steady aggregate while every stream decodes):

| Workload | 1 user | 2 users | 4 users |
|---|---:|---:|---:|
| code (LRU cache module + tests, ends at EOS) | 128 | 188 | 267 |
| prose (400-word story) | 75 | 107 | 163 |

Slowest first token in any cell: 0.1 s. Four users on 512-token replies reach 280-330 tok/s aggregate. Code slows from 512 to 1024 tokens while chat does not: the second half of the code reply is the test file, which drafts worse than the class it tests.

### Cold prefill, one prompt

`tools/prefill_ttft.py` on the serial profile: one fresh random-word prompt per size, streamed, thinking off, prefill rate = prompt tokens / time to the first content token. Receipt: [`evidence/s10-tf066/prefill-serial.txt`](evidence/s10-tf066/prefill-serial.txt).

| Prompt tokens | Time to first token | Prefill |
|---:|---:|---:|
| 13,313 | 5.0 s | 2,654 tok/s |
| 53,584 | 21.0 s | 2,554 tok/s |
| 214,926 | 103.7 s | 2,073 tok/s |

The second rank does not speed prefill up: a single Spark running the Mia'a AI Lab recipe (TensorFold 0.6.1, 2,048-row pieces) reports 2,421 / 2,462 / 2,180 tok/s at 8k / 32k / 128k. The pair's gain is in decode and concurrency, not in the prompt pass; the four-stream cold fill below runs at about 1,900 prompt tok/s in total.

### Four streams at 250k tokens each (1M tokens live on the pair)

`tools/bench_longctx_concurrent.py`: four distinct 250k-token system prompts, four requests started together, greedy, thinking off, 512-token replies. Receipts and the engine's own per-request lines: [`evidence/s10-tf066/longctx-c4/`](evidence/s10-tf066/longctx-c4/).

| Cell | TTFT per stream | Per stream | Aggregate |
|---|---|---:|---:|
| cold: four distinct prompts, 1,001,268 tokens | 181 / 322 / 437 / 528 s | 0.9-1.5 tok/s while the others still fill | about 1,900 prompt tok/s |
| warm: the same prompts cached, prose | 1.7 s each | 29-30 tok/s | **116 tok/s** |
| warm: the same prompts cached, code | 1.6 s each | 52 tok/s (99 rounds for 512 tokens) | **209 tok/s** |
| reference: four streams at 8k, prose | | | 154 tok/s |

Memory is not the limit: with four full windows live the head still had 21 GiB available. Time is. A round with four streams attending over 250k tokens each costs about 100-130 ms against 61 ms at short context, and tokens per round stay what the text allows (2 for prose, 5 for code). Eight short-prompt streams reach 320-520 tok/s and sixteen 650-715; four streams at 1M of context reach 116-209.

Two behaviours to plan around:

- **A long prompt filling starves live replies.** On CUDA `--decode-share` defaults to 0: a prompt pass takes the whole shared round, and a stream that was answering drops to about 1 tok/s until the fill ends. A nonzero share (`EXTRA_ARGS="--decode-share 0.5"`, forwarded to both ranks) sizes the passes so decoding keeps that fraction, at the cost of slower prefill. Not measured here yet.
- **Eight resumable prompts.** The concurrent decoder keeps 8 prompt states (`KEEP = 8`). Four 8k requests between two 250k passes evicted the 250k states, and the next pass re-prefilled all 1M tokens (about 9 minutes). A 1M-token working set means at most eight conversations resume for free.

### Quality: tool calls

`quality/t2.py --tasks tools`: 30 tool prompts, each non-streamed and streamed, exact tool name and exact argument set, judged by the sibling recipe's harness unchanged.

| Serve | Score | Receipts |
|---|---:|---|
| 0.6.6 as shipped (`TOOL_SYSTEM` default) | **60 of 60** | [`tool-system/t2-tools/`](evidence/s10-tf066/tool-system/t2-tools/) |
| 0.6.6 without the tool system message (`TOOL_SYSTEM=`) | 56 of 60 | [`t2-tools/`](evidence/s10-tf066/t2-tools/) |
| base recipe, 0.6.2 | 52 of 60 | its `evidence/s8-quality/` |
| vLLM NVFP4 sibling | 60 of 60 | its evidence |

Without the instruction all four misses are one prompt pair, `t27` and `t47`, streamed and non-streamed: the model calls `translate_text` with the right text and language and adds `"formal": false`. That tool declares `formal` as an optional boolean, so the call is schema-valid; the harness wants the exact key set. The prompt the model sees is byte-identical to vLLM's (same chat template on all three checkpoints), and thinking mode, sampling and drafting do not change the call. These are the MLX 4-bit weights' choice. The 0.6.2 run also lost a prompt to a wrong tool and one to a second unrequested call; neither recurred on 0.6.6.

What does change the call is being told. The recipe's default `TOOL_SYSTEM` ("Tool calls: include only the arguments the user explicitly provided or clearly implied. Never fill in optional arguments with default or guessed values.") is added by the server only to tool requests that carry no system message, so an agent framework with its own system prompt sees no difference. Five phrasings were tried ([`system-default/`](evidence/s10-tf066/system-default/)): two scored 60/60 with every optional-value prompt (`t01`, `t16`, `t28`, `t40`, `t41`) still correct, one over-suppressed and lost `t41`. Set `TOOL_SYSTEM=` in `.env.cluster` to serve without it.

### Quality: the other gates, rerun on 0.6.6

The base recipe's harness (`quality/t2.py`, `tools/run_t3.sh`), same checkpoint, run on this fork's serve (0.6.6, the patch, copy drafts on, `PARALLEL=16`, four workers) on 2026-10-07. Receipts: [`quality/`](evidence/s10-tf066/quality/). The 0.6.2 column is the base recipe's `evidence/s8-quality/`; the vLLM column is the sibling recipe's own run (NVFP4, different weights, a report not a gate).

| Gate | this fork, 0.6.6 | base recipe, 0.6.2 | vLLM NVFP4 sibling |
|---|---:|---:|---:|
| GSM8K-250 (thinking off, greedy) | **95.6%** (239/250) | 95.2% | 95.2% |
| IFEval-120 (strict, prompt level) | **85.0%** (102/120) | 86.7% | 88.3% |
| Tool calls, exact name and arguments (30 × non-streamed + streamed) | **60/60** | 52/60 | 60/60 |
| Repeated word-4-grams over 64 × 512-token replies | **0.04%** (64/64 under the bar) | 0.03% | 0.05% |
| `reasoning_effort` unset / none / low / medium / xhigh | **5/5** | 5/5 | 5/5 |
| Needles, 4k / 16k / 64k / 128k × 3 depths × 2 | **24/24** | 24/24 | 24/24 on a shorter grid |
| Needles at 246-250k prompt tokens × 3 depths × 2 | **6/6** | 6/6 | not run |
| `response_format` json_schema, strict (serial profile, `PARALLEL=1`) | **30/30** | 30/30 | 30/30 |

GSM8K and IFEval moved by 1 and 2 prompts against 0.6.2 (239 against 238, 102 against 104), inside what two greedy runs of the same 4-bit weights on a different engine version do; the engine's own exactness checks compare drafted to plain decoding, not 0.6.2 to 0.6.6. The JSON-schema gate ran on the serial profile (`PARALLEL=1`, grammars are refused under `--parallel` on two ranks), 30/30 with copy drafts on and 48 CUDA graphs captured.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (this cluster: `10.0.0.20` head, `10.0.0.30` worker; stock NVIDIA images use `10.100.8.1` / `10.100.8.2`)
- Docker + NVIDIA Container Toolkit on both nodes, key-based SSH from the head to the worker
- About 115 GB free disk per node for the weights (113.2 GB snapshot), plus 24.5 GB for the image
- Exclusive GPUs. `run.sh` refuses to start next to another `--gpus all` container (a `buildx_buildkit_*` builder is allowed).

```bash
hf auth login      # or: export HF_TOKEN=hf_...
```

## Quick start

On the head Spark:

```bash
git clone https://github.com/BobClawblaw/qwen38-flashnext-tensorfold-2x-dgx-sparks.git qwen38-flashnext-tensorfold
cd qwen38-flashnext-tensorfold
$EDITOR .env.cluster        # HEAD_IP, WORKER_HOST=user@worker, PORT, KV_DTYPE, MTP_DRAFTS, HCA
VALIDATE_ONLY=1 ./run.sh    # checks the settings, no Docker
PROFILE=concurrent ./run.sh # sixteen streams; plain ./run.sh serves one at a time (and response_format)
```

The head checks that the weights are complete on its disk, builds the image when it is missing (about 2 minutes on top of the base image, which it pulls once), copies it to the worker when the worker's image ID differs (24.5 GB over the link), starts rank 1 on the worker over SSH, then rank 0, and waits for `/health` and `/v1/models`. A first start of a new engine commit JIT-compiles the CUDA kernels (several minutes); later starts load in about 45 s.

Autostart on boot:

```bash
sudo cp qwen38-tensorfold.service /etc/systemd/system/
sudoedit /etc/systemd/system/qwen38-tensorfold.service   # User=, paths
sudo systemctl daemon-reload && sudo systemctl enable --now qwen38-tensorfold
```

`systemctl stop qwen38-tensorfold` (or `./stop.sh`) stops rank 0 first with SIGTERM, then waits for rank 1 and removes both containers. Do not run `./run.sh` by hand while the unit is active: two orchestrators, one pair.

Smoke test (thinking is off by default):

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Probes and gates against the live API (`--url http://127.0.0.1:8888/v1/chat/completions` on this cluster; the scripts default to port 8000):

```bash
python3 smoke_thinking.py --model TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP --url http://127.0.0.1:8888/v1/chat/completions
python3 smoke_tools.py    --model TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP --url http://127.0.0.1:8888/v1/chat/completions
python3 bench_decode.py   --model TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP --url http://127.0.0.1:8888/v1/chat/completions
python3 quality/t2.py --tasks tools --url http://127.0.0.1:8888 --out /tmp/t2-tools
```

## Defaults

`recipe.yaml` is the source of truth for the shipped defaults. Edit it, then run `python3 kit/render.py`. CI fails when this table or the `run.sh` block drifts from it. `.env.cluster` overrides them per cluster; this cluster's file sets `PORT=8888`, `KV_DTYPE=int8`, `MTP_DRAFTS=15`, `HCA=rocep1s0f1`.

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Engine | TensorFold `cb2ebf0540f42604e2759b2ddef497861e928248` (v0.6.6), built into `tf-qwen38-flashnext:0.6.6` from `docker/Dockerfile` |
| Patch | `patches/flashnext-tools-0.6.6.patch`: `--tool-system` on the CUDA server, and a tool call keeps only the parameters the offered tool declares (`docker/patches/README.md`) |
| Tool system message | `Tool calls: include only the arguments the user explicitly provided or clearly implied. Never fill in optional arguments with default or guessed values.` (added by the server to a chat request that offers tools and has no system message; `TOOL_SYSTEM=` serves none) |
| Model | `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` at `2b170fa6309d5d1ee380b35636075fac7945f286` |
| Ranks | `--tp 2`: rank 1 on `spark2` first, then rank 0 (HTTP) on the head; rendezvous `10.100.8.1:29551` |
| `--context` | 262144 (the native window, on both ranks) |
| `--kv-dtype` | `bf16` |
| MTP drafts | `--mtp-drafts 15 --mtp-confidence 0.70` |
| `--parallel` | 1 by default; `PROFILE=concurrent` (or `PARALLEL=16`) serves sixteen streams on two ranks (TensorFold 0.6.4+) |
| Default thinking | off (`THINKING=0`); requests override with `chat_template_kwargs.enable_thinking` |
| `--max-tokens` | 32768 (the reply cap when a request sets none) |
| NCCL | `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1` |
| API | `http://<head>:8000/v1`, served as `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` |
| Container | `tf-qwen38-flashnext` |
<!-- END generated defaults -->

`run.sh` refuses, before `VALIDATE_ONLY` exits:

- non-decimal or zero-padded integers, `MTP_CONFIDENCE` outside [0, 1], a `TF_SHA` / `SNAPSHOT_SHA` that is not 40 hex
- `TP` other than 1 or 2, `CONTEXT` above the native 262144 (TensorFold serves no YaRN), `MTP_DRAFTS` above the engine cap 15, `KV_DTYPE` other than bf16 / int8 / int4
- a `TF_PATCH` that is not `none` or a pinned file under `docker/patches/`, or whose sha256 differs from its pin
- `EXTRA_ARGS` that re-sets a flag `run.sh` builds (`--tp`, `--rank`, `--master*`, `--host`, `--port`, `--name`, `--context`, `--kv-dtype`, `--mtp-*`, `--parallel`, `--thinking`, `--max-tokens`, `--no-drafts`), `--vision*` (one GPU only), and `--prefill-fp8` / `--drafter` / `--ple-on-ssd` / `--ssd-experts` (refused or unused for this checkpoint)
- an `EXTRA_ENV` word that is not `KEY=VALUE`

At start it also refuses another running GPU container on either node, a busy port, an image whose labels do not match `TF_SHA` / the patch sha, and an incomplete snapshot when `SKIP_DOWNLOAD=1`. `BENCH_ONLY=1` binds the API to 127.0.0.1. `MTP_DRAFTS=0` serves without drafts.

## Not supported (TensorFold 0.6.6 on two ranks)

- With `--parallel` above 1: no `response_format` / `guided_*` grammars (HTTP 400), no logprobs, no images. The serial default serves `response_format` through xgrammar on both ranks.
- Each request decodes to `max_tokens` or EOS on both ranks. A client disconnect, a stop string, a forced `tool_choice` and a `thinking_budget` cut stop what is sent, not the GPU work. Set `max_tokens` per request; `MAX_TOKENS` (default 32768, clamped by the engine to the room left in the window) applies when a request sets none. The base recipe's 4096 let a thinking reply run out inside its think block and return nothing.
- A rank that dies mid-request leaves the other waiting in NCCL with no timeout, and `/health` on rank 0 does not check rank 1. Restart with `systemctl restart qwen38-tensorfold` (or `./stop.sh && ./run.sh`).
- No `n > 1`, no `/tokenize` at two ranks, no presence/frequency penalties (ignored). The reasoning field is `reasoning_content`. The default seed is a hash of the prompt, so identical sampled requests repeat unless they carry a `seed`.

## Environment

`.env.cluster` (this cluster's values; `user@` is the worker's SSH login):

```bash
HEAD_IP=10.0.0.20
WORKER_HOST=user@10.0.0.30
PORT=8888
KV_DTYPE=int8
MTP_DRAFTS=15
HCA=rocep1s0f1
```

Everything in the Defaults table can go there (`IFACE`, `CONTEXT`, `MAX_TOKENS`, `PARALLEL`, `MEMORY_RESERVE_GIB`, …). A variable already in the environment wins over the file; `ENV_CLUSTER=/dev/null` ignores it (tests and CI do).

Pin `NCCL_IB_HCA`: GB10 exposes four HCAs and two of them are DOWN; unpinned NCCL can pick a dead one. `TF_CACHE` holds the kernel builds per engine commit (default `~/.cache/tensorfold-qwen38/<TF_SHA>`). `MEMORY_RESERVE_GIB` sets TensorFold's startup reserve. Memory per rank at the full window with int8 KV: TensorFold's startup estimate is 45.6 GiB on the GPU plus the 29.8 GiB n-gram tables mlocked in host memory (`--ulimit memlock` and `IPC_LOCK` are passed for that). Read unified memory with `free -h`, not `nvidia-smi`.

The NGC base image sets `NCCL_NET_PLUGIN=spcx`; NCCL logs that the Spectrum-X plugin is unsupported on these HCAs and falls back to its own RoCE transport. The warning is cosmetic; one 1,500-token reply moves about 1.9 GB over the link.

## Logs

```bash
docker logs -f tf-qwen38-flashnext
ssh user@10.0.0.30 docker logs -f tf-qwen38-flashnext
journalctl -u qwen38-tensorfold -b
```

TensorFold prints one `done req-…` line per request: tokens, tok/s, TTFT, prefill time, verify rounds, accepted drafts and the reply's token sha. `GET /health` carries cumulative counters and the live decode and prefill speed; `GET /metrics` is Prometheus with the `tensorfold:` prefix.

## Evidence

Every number in this README has a file under [`evidence/`](evidence/). This fork's session is [`s10-tf066`](evidence/s10-tf066/); `s1` to `s9` are the base recipe's (TensorFold 0.6.2) and are kept as its receipts. `recipe.yaml` names the file per measured row; `python3 kit/render.py --check` lists the rows that still have none.

## Gotchas

- TensorFold opens HTTP only after the model is loaded, so "connection refused" means still loading.
- The first request after a boot is slower than the rest (prompt warm-up in the concurrent decoder); the tables are from repeated runs on the same boot.
- `docker stop` on a rank without `--init` would wait out its timeout: TensorFold's rank 1 installs no SIGTERM handler. `run.sh` passes `--init`.
- A buildx builder container (`buildx_buildkit_*`) is exempt from the foreign-GPU-container check; anything else holding the GPU makes `run.sh` refuse to start.
- The `TensorFold/` repo id prints a cosmetic "untested" note at start (TensorFold lists the checkpoint under its old `Vontra/` name).

## Credits

- Base recipe: [sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark](https://github.com/sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark) (MIT): `run.sh`, `stop.sh`, `kit/`, `tests/`, `tools/`, `quality/`, the smokes, `bench_decode.py` and `evidence/s1` to `s9`.
- Engine: [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0 from 0.6.0; earlier code MIT), tag v0.6.6. Two-rank `--parallel` is upstream's (#141 by BHCC2025 and ashhart). `tools/vendor/bench_concurrent.py` is TensorFold's, unmodified.
- Checkpoint: [TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP); base model Qwen3.8-Flash-Next.
- Harness: `bench_decode.py`, the smokes and `quality/` come from the vLLM sibling recipe ([sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark)).

## License

The recipe's own files are MIT ([`LICENSE`](LICENSE)). `docker/patches/flashnext-tools-0.6.6.patch` changes TensorFold source: Apache-2.0, provenance and changes in [`docker/patches/README.md`](docker/patches/README.md). `tools/vendor/bench_concurrent.py` is TensorFold's, unmodified: MIT and Apache-2.0, per its header. Upstream's notice is in [`NOTICE`](NOTICE) and the license texts are in [`LICENSES/`](LICENSES/). Model weights follow the source model license on Hugging Face.
