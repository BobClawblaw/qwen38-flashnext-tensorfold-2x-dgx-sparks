# s10 · image and video input on two ranks (patch part 4), 2026-10-07

Image `tf-qwen38-flashnext:0.6.6` with `flashnext-tools-0.6.6.patch` (four parts), `PROFILE=concurrent` (`PARALLEL=16`), `VISION=1`, int8 KV, MTP 15 at 0.70, one RoCE HCA.

| File | What |
|---|---|
| `startup.txt` | both ranks' vision and engine lines: the tower on rank 0 (0.84 GiB + 4 GiB workspace, 53.4 GiB free for caches) and rank 1 attaching what rank 0 encodes (58.0 GiB free) |
| `visioncheck.txt` | `tools/visioncheck.py`: a drawn PNG (red circle, blue square), a 3840 x 2160 photo (Ubuntu's "Clouds" wallpaper, 4,101 prompt tokens) and a 4 s 320 x 240 video made with ffmpeg (green frame turning red, a white square moving right) |
| `exact.txt` | `tools/vision_exact.py`: drafted against `"draft": false` on four image prompts, four image requests together against alone, a text request beside them against alone; every reply token sha equal |
| `ruler-vision-on.out` | `bench_decode.py --concurrency 1 2 --phase both` with vision on: prose 66.2 / 60.8, structured 231.2 / 208.6 (vision off, `../bench-frozen-2.out`: 65.6 / 60.3, 232.9 / 210.7) |
| `text-prefill-4k.txt` | `tools/prefill_ttft.py --words 1850`: 4,124 text tokens prefill in 1.69 s, against 3.2 s for the 4,101-token photo prompt (the tower's share, about 1.5 s) |
| `ab-*.out`, `ab.log` | same host session, `VISION=0` then `VISION=1` restarts of the concurrent profile: `tools/bench_openai.py` 92.4 / 95.2 -> 89.3 / 94.4 tok/s, `bench_decode.py` prose 66.2 -> 65.2, structured 231.3 -> 232.0, `tools/prefill_cold.py` 2k-64k cold prompts equal within 1% (2,411 / 2,513 / 2,494 / 2,467 / 2,371 against 2,401 / 2,514 / 2,519 / 2,475 / 2,382 tok/s) |
| `pytest-patched.txt`, `pytest-pristine.txt` | upstream's full test suite inside the image: the patched tree 4111 passed / 127 failed / 11 errors / 175 skipped, pristine cb2ebf0 4090 / 127 / 11 / 175, the same 138 failing tests in both (upstream's own, Metal/MLX and network-bound); the 21 extra passes are the patch's tests |

The first systemd start after the switch hit the known NCCL init race (rank 1's connection closed during the rendezvous, `RuntimeError: NCCL error 6`); the unit's `Restart=on-failure` brought the pair up 30 s later. Not vision-related: the same happened on 2026-10-06 switching profiles.
