# docker/patches/

Patches the image build applies on top of TensorFold at `TF_SHA` (`docker build --build-arg TF_PATCH=patches/<file> --build-arg TF_PATCH_SHA=<sha256>`). Each shipped patch's sha256 is pinned in `run.sh` (`PATCH_PINS`) and `recipe.yaml` (`engine.patches`), and `tests/` checks the files against the pin. `run.sh` refuses a patch file that differs from its pin. The build checks the copied file against the sha it is given, and `run.sh` refuses an image whose `tensorfold.patch_sha` label differs. A regenerated patch therefore needs a new pin, and new evidence for the numbers it affects.

## flashnext-tools-0.6.6.patch

The default image carries it. Two changes to TensorFold v0.6.6 (`cb2ebf0540f42604e2759b2ddef497861e928248`), Apache-2.0 as TensorFold; nine files, including two test files.

### 1. `--tool-system TEXT` on the CUDA server

A system message the server adds to a chat request that offers tools and carries no system message of its own. A request's own system message wins, and a request without tools is rendered exactly as before. Files: `src/tensorfold/cli_args.py` (the flag), `src/tensorfold/cli.py` (plumbed to the app), `src/tensorfold/cuda/server.py` (`App.__init__` keeps it; `_prepare` prepends it before the template renders), `tests/test_cuda_tool_system.py`.

- **Why.** Qwen3.8-Flash-Next (MLX 4-bit) fills a declared optional parameter it was not asked for on 2 of the quality harness's 30 tool prompts (`formal: false` on `translate_text`), streamed and non-streamed: 56/60 on the exact-arguments check, where the vLLM NVFP4 sibling scored 60/60. The prompt the model sees is otherwise identical to vLLM's (same chat template, byte for byte), thinking mode and sampling do not change the call, and no engine setting reaches it. An instruction does: with the recipe's default text (`TOOL_SYSTEM` in `recipe.yaml`) the full 60 pass, and every prompt that needs an optional value still gets it. The search over five phrasings is in `evidence/s10-tf066/system-default/` (two of five scored 60/60, one over-suppressed and lost `t41`).
- **Scope.** Only chat requests that offer tools and send no system message. Agents that ship their own system prompt are untouched; so is plain text. `TOOL_SYSTEM=` (empty) serves without it.

### 2. A tool call keeps only the parameters the offered tool declares

Files: `src/tensorfold/tool_parameters.py` (`keeps_parameter`, the one rule), `src/tensorfold/server/tools.py` (the end parser), `src/tensorfold/cuda/reply_text.py` (the CUDA reply parser), `src/tensorfold/engine/tool_draft.py` (`ToolCallStreamer` consumes an undeclared parameter without emitting a delta, so streamed and non-streamed arguments agree), `tests/test_tool_undeclared_arguments.py`. A tool that declares no properties still takes every argument; JSON-shaped and GLM calls are not filtered. This does not touch the harness's `formal` case (that parameter is declared); it covers a parameter the tool has no slot for, which a client's tool runner would otherwise receive.

### Validation

`tests/test_cuda_tool_system.py` and `tests/test_tool_undeclared_arguments.py`, plus upstream's tool and CUDA-server tests, on the host (the one upstream test that needs `mlx` fails on Linux with or without the patch). On the pair: `evidence/s10-tf066/tool-system/` (the 60-call harness through the served flag) and the README's Quality section.

To change the patch: edit a clean v0.6.6 checkout, run the tests above, regenerate with `git add -N <new files>; git diff HEAD > flashnext-tools-0.6.6.patch`, update the pin in `run.sh` and `recipe.yaml`, rebuild the image, and re-run `quality/t2.py --tasks tools`. Never edit the patch file by hand.

## Retired: pr141-on-0.6.2.patch

Up to the 0.6.2 image, `PROFILE=concurrent` built a second image with upstream PR #141 (`--parallel N` on two ranks) ported onto 0.6.2. TensorFold 0.6.4 merged that work (release notes: "Flash Next on two DGX Sparks serves concurrent requests"), so the port is gone from this tree. Its history and validation stay in the base recipe's [`evidence/s6-pr141/`](../../evidence/s6-pr141/).
