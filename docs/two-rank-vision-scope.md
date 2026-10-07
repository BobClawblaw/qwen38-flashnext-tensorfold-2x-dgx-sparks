# Scope: image input on two ranks

Status: scoped, not started (2026-10-07). TensorFold serves Flash Next images on one GPU with `--parallel 2` or more and refuses `--vision` at `--tp 2` (`families/qwen4_exp/cuda/engine.py`, the `vision and (streams < 2 or tp != 1)` check). This note says what a patch needs to lift that, in the style of the recipe's other patch parts.

## How images work on one GPU today

1. The HTTP server (rank 0) decodes the request's images with the tower's frontend (`server/prompts.py: prepare_images`) and renders the prompt; the result is the token ids plus a `PreparedVisionPrompt` (pixel patches, `image_grid_thw`, the 3-axis rotary positions, `rope_delta`).
2. `MultiDecoder.admit` refuses prefix reuse for image prompts (`_slot_for(prompt, s.draft and s.vision is None)`), then `image_rows.begin` runs the tower on the admitted slot: `tower.encode(prepared, prompt)` gives the image features (one row per image token, bf16, hidden size 2560), the feature row indices and the positions, and `attach` stores them on the stream's state.
3. During the prompt passes `image_rows.embed` writes the features over the placeholder rows of each prompt piece; `finish` drops the features after prefill, the rotary positions stay with the stream. Image streams keep no prompt states (`stops` are empty) and never take the lone-stream graph slot.

## What two ranks need

Both ranks must feed bit-identical hidden states into their halves of every layer. Text does that by construction (both embed the same ids). For images the features must be identical on both ranks, so they are computed once and sent, not recomputed.

- **Rank 0 only holds the tower.** `capacity_geometry(..., vision, rank, workspace)` and `vision_weights(..., vision, rank)` already take the rank, so the geometry and weight plumbing expect this. Rank 1 reserves no tower and no vision workspace. Engine gate: allow `vision` at `tp == 2` when `streams >= 2`; keep refusing it on the serial engine.
- **Admission message.** `multi_tp._prepare_admission` sends `["admit", sid, prompt, count, sampling, draft, stop_eos, background, plan]` over the control link (JSON, TCP). Add a `vision` entry: `image_grid_thw`, the feature row indices, `rope_delta`, and the feature tensor's shape. Rank 1 rebuilds the positions on the CPU from the token ids and the grids (`vision/qwen_processing.image_positions`, deterministic) rather than receiving the 3 x length positions array. Add `has_vision` to the `_agree` digest so both ranks plan the same admission (no prefix reuse, same slot growth).
- **Feature transfer.** Right after the admission agrees, rank 0 sends the feature tensor and rank 1 receives it with `comm.exchange` (the point-to-point primitive the two-rank engines already have, `cuda/comm.py`). Both ranks reach this call in lockstep because admission already is a lockstep point. Size: at most the request's visual-token budget (16,384 tokens by default) x 2560 x 2 bytes, 80 MB worst case, a few milliseconds on the RoCE link.
- **Attach on rank 1.** `image_rows.attach(st, Encoded(features=received, rows, positions, rope_delta), len(prompt))`; nothing else in the prompt passes or the rounds changes, `embed` and the positions kernel are per-rank local reads of identical data.
- **Memory.** Rank 0 carries the tower (about 5 GiB) and its workspace; `--vision-offload` keeps the tower in host RAM between images. The admission plan's `ready()` already checks both ranks' room, so the asymmetry is handled; the recipe would document that rank 0 keeps about 5 GiB less for stream caches.
- **Not in scope.** Video (CUDA vision upstream takes images only; video is a single-Spark recipe patch), logprobs, grammars with images, prefix reuse for image prompts (not available on one GPU either).

## Files

`families/qwen4_exp/cuda/engine.py` (gate, tower on rank 0, startup line), `multi.py` (`admit`: drop the two-rank refusal for vision), `multi_tp.py` (message fields, digest, the exchange), `multi_plan.py` (`admission`: no prefix reuse when the stream carries images; `ready`: unchanged), `image_rows.py` (an `Encoded` built from a received tensor), `cuda/server.py` (nothing; rank 0 serves HTTP as today). About 150 to 250 lines plus tests.

## Tests and receipts

- Host: the admission message round-trips the vision fields; the digest differs when one rank has images and the other does not; positions rebuilt on rank 1 equal rank 0's for the vision probes' prompts.
- Pair, exactness: the same image request served at `--tp 2` and at `--tp 1 --parallel 2` on one Spark, token sha equal; concurrent against solo on two ranks; `"draft": false` against drafted. The single-Spark recipe's `vision_probes/probes.py` (images with fixed descriptions) is the correctness set.
- Pair, speed: a text ruler before and after (nothing should move), and image requests at 1 and 4 users with the time the tower adds to TTFT.

## Effort and risk

Two to three days of work plus the measurement session. The risks are ordering of the feature exchange against the round loop's other collectives (it must happen inside the admission lockstep, before the next round plan), rank 0's smaller cache room, and keeping the plan digest honest when a request is refused on one rank (an oversized image on rank 0 must refuse before rank 1 is told, as grammar errors do today). Upstream's Zig engine has no Flash Next on CUDA yet, so this would live in the recipe's patch and be offered as a reference, like the copy drafts.
