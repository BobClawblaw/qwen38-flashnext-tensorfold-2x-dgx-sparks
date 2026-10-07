# docker/patches/

Patches the image build applies on top of TensorFold at `TF_SHA` (`docker build --build-arg TF_PATCH=patches/<file> --build-arg TF_PATCH_SHA=<sha256>`). Each shipped patch's sha256 is pinned in `run.sh` (`PATCH_PINS`) and `recipe.yaml` (`engine.patches`), and `tests/` checks the files against the pin. `run.sh` refuses a patch file that differs from its pin. The build checks the copied file against the sha it is given, and `run.sh` refuses an image whose `tensorfold.patch_sha` label differs. A regenerated patch therefore needs a new pin, and new evidence for the numbers it affects.

## flashnext-tools-0.6.6.patch

The default image carries it. Three changes to TensorFold v0.6.6 (`cb2ebf0540f42604e2759b2ddef497861e928248`), Apache-2.0 as TensorFold; sixteen files, including three test files.

### 1. `--tool-system TEXT` on the CUDA server

A system message the server adds to a chat request that offers tools and carries no system message of its own. A request's own system message wins, and a request without tools is rendered exactly as before. Files: `src/tensorfold/cli_args.py` (the flag), `src/tensorfold/cli.py` (plumbed to the app), `src/tensorfold/cuda/server.py` (`App.__init__` keeps it; `_prepare` prepends it before the template renders), `tests/test_cuda_tool_system.py`.

- **Why.** Qwen3.8-Flash-Next (MLX 4-bit) fills a declared optional parameter it was not asked for on 2 of the quality harness's 30 tool prompts (`formal: false` on `translate_text`), streamed and non-streamed: 56/60 on the exact-arguments check, where the vLLM NVFP4 sibling scored 60/60. The prompt the model sees is otherwise identical to vLLM's (same chat template, byte for byte), thinking mode and sampling do not change the call, and no engine setting reaches it. An instruction does: with the recipe's default text (`TOOL_SYSTEM` in `recipe.yaml`) the full 60 pass, and every prompt that needs an optional value still gets it. The search over five phrasings is in `evidence/s10-tf066/system-default/` (two of five scored 60/60, one over-suppressed and lost `t41`).
- **Scope.** Only chat requests that offer tools and send no system message. Agents that ship their own system prompt are untouched; so is plain text. `TOOL_SYSTEM=` (empty) serves without it.

### 2. A tool call keeps only the parameters the offered tool declares

Files: `src/tensorfold/tool_parameters.py` (`keeps_parameter`, the one rule), `src/tensorfold/server/tools.py` (the end parser), `src/tensorfold/cuda/reply_text.py` (the CUDA reply parser), `src/tensorfold/engine/tool_draft.py` (`ToolCallStreamer` consumes an undeclared parameter without emitting a delta, so streamed and non-streamed arguments agree), `tests/test_tool_undeclared_arguments.py`. A tool that declares no properties still takes every argument; JSON-shaped and GLM calls are not filtered. This does not touch the harness's `formal` case (that parameter is declared); it covers a parameter the tool has no slot for, which a client's tool runner would otherwise receive.

### 3. Copy drafts for Flash Next on CUDA

New `src/tensorfold/families/qwen4_exp/cuda/copy_drafts.py` (Nemotron's `CopyIndex`, unchanged in substance) and edits to `decode.py` (the serial MTP loop), `multi.py` (`_draft_all`, the shared rounds), `multi_solo.py` (the lone stream's graph slot), `multi_fill.py` (the first drafts after a prompt) and `engine.py` (the switch and the startup line); `tests/test_copy_drafts.py`.

- **What.** When a reply's last 8 tokens repeat text seen earlier in the prompt or the reply, the round drafts that text's continuation, whole, up to the depth (15), instead of asking the MTP head. Otherwise the MTP chain runs as before. The verify step is untouched, so replies stay byte-identical to one-token decoding; the kept rows still go into the MTP cache, so the head resumes cleanly on the next round that needs it.
- **Why.** Quoting, editing and refactoring copy their input. The MTP head landed 6.2-6.6 tokens a round on such replies here (its confidence floor cuts the chain), where a certain chain of 15 lands up to 14 (counting measured 13.6). Fresh prose and code are untouched: no earlier copy, no copy draft.
- **Two ranks.** Both ranks hold the same tokens (sampling is gathered), so both build the same index and propose the same chain; the per-round plan digest (`multi_tp.shape`) includes each stream's drafts and would stop the ranks if they ever differed. In the shared rounds the first chain level's picks still cover every absorbed stream (the two-rank gather reads them in order); a copying stream's pick is dropped and it joins no later level.
- **Cost.** The index is a dict of 8-gram starts: a 250k-token prompt builds in 0.12 s on the host at admission, a chain lookup is microseconds, and each stream's index goes with it.
- **Switch.** On by default with MTP drafts; `TENSORFOLD_COPY_DRAFTS=0` (an `EXTRA_ENV` word) turns it off. Off without MTP drafts (`MTP_DRAFTS=0`), and off for grammar-constrained replies.

### Validation

`tests/test_cuda_tool_system.py`, `tests/test_tool_undeclared_arguments.py` and `tests/test_copy_drafts.py`, plus upstream's tool, CUDA-server and concurrent-decoder tests (`test_cuda_batch_admit.py`, `test_cuda_growing_caches.py`), inside the image on the CPU and on the host (the one upstream test that needs `mlx` fails on Linux with or without the patch). On the pair: `evidence/s10-tf066/tool-system/` (the 60-call harness through the served flag), `evidence/s10-tf066/copy-drafts/` (copy cells and the concurrency ruler against `"draft": false`), and the README's Quality section.

To change the patch: edit a clean v0.6.6 checkout, run the tests above, regenerate with `git add -N <new files>; git diff HEAD > flashnext-tools-0.6.6.patch`, update the pin in `run.sh` and `recipe.yaml`, rebuild the image, and re-run `quality/t2.py --tasks tools`. Never edit the patch file by hand.

## Retired: pr141-on-0.6.2.patch

Up to the 0.6.2 image, `PROFILE=concurrent` built a second image with upstream PR #141 (`--parallel N` on two ranks) ported onto 0.6.2. TensorFold 0.6.4 merged that work (release notes: "Flash Next on two DGX Sparks serves concurrent requests"), so the port is gone from this tree. Its history and validation stay in the base recipe's [`evidence/s6-pr141/`](../../evidence/s6-pr141/).
