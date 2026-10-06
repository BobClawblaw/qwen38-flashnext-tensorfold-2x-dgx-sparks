# s10 · four streams at 250k tokens each (1M tokens live on the pair)

`tools/bench_longctx_concurrent.py` on the 0.6.6 serve (`PARALLEL=8`, int8 KV). Four distinct 250k-token fillers as system
messages, four requests started together, greedy, thinking off, streamed. `engine-done-lines.txt` has TensorFold's own
per-request lines (prompt, cached, tok/s, rounds, accepted) for every request below: those are the ground truth.

| File | Cell | Result |
|---|---|---|
| `c4-8k.out` | 4 × 8k, cold, 256 tokens, prose | TTFT 3.9 / 7.8 / 10.3 / 12.8 s; prompts fill inside the decode rounds |
| `c4-250k.out` | 4 × 250k, cold, 256 tokens, prose | TTFT 181 / 322 / 437 / 528 s; a stream decoding while others fill: 0.9-1.5 tok/s; the last one alone: 19.9 chunks/s |
| `c4-250k-warm.out` | same prompts, cached, 512 tokens, prose | TTFT 1.7 s each; **116 tok/s aggregate** steady, about 29-30 tok/s per stream |
| `c4-250k-warm-code.out` | same prompts, cached, 512 tokens, code | TTFT 1.6 s each; **209 tok/s aggregate**, 52 tok/s per stream (engine: tok/s=52.2, 99 rounds, 412/559 drafts accepted) |
| `c4-8k-warm.out` | 4 × 8k, 512 tokens, prose (prompts were no longer cached) | 154 tok/s steady while all four decoded |

Caveat on the `STREAM` lines in the `.out` files of this session: their `decode_tok_s` counted stream chunks, not tokens
(a chunk carries several tokens under MTP). Multiply by `completion_tokens / chunks` for tokens per second: prose 512/258 ≈ 2.0,
code 512/98 ≈ 5.2. The `SUMMARY` aggregates were computed from token counts and are right. The script was fixed after
these runs (`decode_tok_s` now scales by tokens per chunk).

Two things the session showed:
- `--decode-share` defaults to 0 on CUDA: a prompt pass takes the whole shared round, so live replies crawl while a long
  prompt fills. A nonzero share (allowed through `EXTRA_ARGS`, forwarded to both ranks) trades prefill speed for that.
- The concurrent decoder keeps 8 prompt states (`KEEP = 8` in `families/qwen4_exp/cuda/engine.py`). Four 8k requests
  between two 250k passes evicted the 250k states, and the next pass re-prefilled all 1M tokens.
