# s10 · a default system instruction for tool requests

The quality harness's two tool-call misses (`t27`, `t47`: `translate_text` called with the right text and language plus an
unrequested `"formal": false`) do not move with thinking mode, sampling, drafting or the engine; the chat template is
byte-identical to the vLLM sibling's. They do move with an instruction. `search.txt`: five system messages, each run through
the full 60-call harness (30 prompts, non-streamed and streamed) with the message prepended client-side; the judge is
`quality/t2.py`'s unchanged. `t2-tools-<letter>.json` has every call.

| | Text | Score |
|---|---|---:|
| A | When you call a tool, set only the parameters the user's request requires. Leave optional parameters out unless the user asked for them. | fixes t27, not t47 (6-prompt probe) |
| B | Tool calls: include only the arguments the user explicitly provided or clearly implied. Never fill in optional arguments with default or guessed values. | **60/60** |
| C | When calling a function, pass only the required parameters plus any optional parameter the user explicitly mentioned. Do not add optional parameters on your own. | 58/60 (loses t41, which needs `keep_aspect: true`) |
| D | Function calling rules: 1) Always include every required parameter. 2) Include an optional parameter only when the user states its value. 3) Never guess, infer or default an optional parameter; omit it instead. | **60/60** |
| E | Be minimal in tool calls: omit any optional argument the user did not specify. | fixes t27, not t47 (6-prompt probe) |

B is the recipe's default `TOOL_SYSTEM`. The server adds it (`--tool-system`, from `docker/patches/flashnext-tools-0.6.6.patch`)
only to chat requests that offer tools and carry no system message. `../tool-system/` has the harness run through the served
flag rather than client-side injection.
