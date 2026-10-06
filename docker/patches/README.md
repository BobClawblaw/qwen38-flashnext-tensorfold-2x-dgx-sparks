# docker/patches/

Patches the image build applies on top of TensorFold at `TF_SHA` (`docker build --build-arg TF_PATCH=patches/<file> --build-arg TF_PATCH_SHA=<sha256>`). Each shipped patch's sha256 is pinned in `run.sh` (`PATCH_PINS`) and `recipe.yaml` (`engine.patches`), and `tests/` checks the files against the pin. `run.sh` refuses a patch file that differs from its pin. The build checks the copied file against the sha it is given, and `run.sh` refuses an image whose `tensorfold.patch_sha` label differs. A regenerated patch therefore needs a new pin, and new evidence for the numbers it affects.

## undeclared-tool-args-0.6.6.patch

The default image carries it. A Qwen XML tool call (`<function=name><parameter=key>…`) keeps only the parameters the offered tool declares in its JSON schema. A tool that declares no properties still takes every argument.

- **Why.** Qwen3.8-Flash-Next (MLX 4-bit) sometimes writes a parameter the offered tool does not declare, and TensorFold typed and forwarded it, so a client's tool runner saw an argument its function has no slot for. Dropping the undeclared key leaves the call the client asked for. Note what this does not do: the quality harness's two `translate_text` misses carry `formal: false`, which that tool *does* declare as an optional boolean, so they are kept and still count as misses there (the harness wants the exact key set). See the README's Quality section for the measured count.
- **Base.** TensorFold v0.6.6, `cb2ebf0540f42604e2759b2ddef497861e928248`.
- **License.** Apache-2.0, as TensorFold (its `LICENSE` and `NOTICE` apply to the patched files).
- **The file.** `git diff` of a v0.6.6 checkout with five files touched:
  - `src/tensorfold/tool_parameters.py`: `keeps_parameter(key, properties)`, the one rule.
  - `src/tensorfold/server/tools.py`: the end parser (`_parse_tool_call_payload`) skips an undeclared key.
  - `src/tensorfold/cuda/reply_text.py`: the CUDA reply parser skips it.
  - `src/tensorfold/engine/tool_draft.py`: `ToolCallStreamer` consumes an undeclared parameter without emitting a delta for it, so streamed arguments equal the non-streamed ones.
  - `tests/test_tool_undeclared_arguments.py`: the rule on all three parsers, with the undeclared parameter first, last and alone.
- **Scope.** JSON-shaped tool calls (`{"name": …, "arguments": {…}}`) and GLM calls are not filtered; this checkpoint emits the XML form. The JSON path keeps upstream behaviour.
- **Validation.** `tests/test_tool_undeclared_arguments.py` and upstream's tool tests (`tests/test_tool_*.py`, `tests/test_cuda_tool_*.py`, `tests/test_glm_tool_calls.py`): 235 passed on the host; `tests/test_history_after_tool.py` has one test that needs `mlx` and fails on Linux with or without the patch. On the pair: see the README's Quality section and `evidence/`.

To change the patch: edit a clean v0.6.6 checkout, run the tests above, regenerate with `git add -N <new files>; git diff > undeclared-tool-args-0.6.6.patch`, update the pin in `run.sh` and `recipe.yaml`, rebuild the image, and re-run `quality/t2.py --tasks tools`. Never edit the patch file by hand.

## Retired: pr141-on-0.6.2.patch

Up to the 0.6.2 image, `PROFILE=concurrent` built a second image with upstream PR #141 (`--parallel N` on two ranks) ported onto 0.6.2. TensorFold 0.6.4 merged that work (release notes: "Flash Next on two DGX Sparks serves concurrent requests"), so the port is gone from this tree. Its history and validation stay in the base recipe's [`evidence/s6-pr141/`](../../evidence/s6-pr141/).
