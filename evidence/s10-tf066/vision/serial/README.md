# s10 · image prompts on the serial profile (patch part 5), 2026-10-07

`PROFILE=serial` (`PARALLEL=1`), `VISION=1`, same image as the parent directory's part 4 plus part 5 (the serial engine serves image prompts; `vision_ranks.py` carries the features between the ranks for both engines). Same host session for both halves of the A/B: `VISION=0` first (`ab-vision-off.log`, `ab-serial-*-v0.out`), then `VISION=1` (`vision-on.log`, `ab-serial-*-v1.out`). The `VISION=1` start was refused once because a test-suite container was still running (run.sh's exclusive-GPU check), and the unit's retry started it 30 s later.

| Cell | `VISION=0` | `VISION=1` |
|---|---:|---:|
| `tools/bench_openai.py`, fibonacci-raw / gpu-chat-no-think | 101.6 / 101.6 tok/s | 100.7 / 99.2 tok/s |
| `bench_decode.py` c=1, prose / structured | 72.3 / 250.2 | 72.1 / 239.0 |
| `tools/prefill_cold.py` 2k / 8k / 16k / 32k / 64k, prompt tok/s | 2,413 / 2,522 / 2,534 / 2,479 / 2,400 | 2,420 / 2,545 / 2,550 / 2,509 / 2,404 |

Image probes with vision on (`visioncheck.txt`): the drawn PNG, the photo and the video answered as on the concurrent profile, with the same reply token shas (`a7588cf7f668`, `e99c86a30d35`, `333c343e9195`); the 4,101-token photo prompt passes in 1.64 s on the serial engine (3.2 s on the concurrent decoder's 2,048-row pieces). `exact.txt`: drafted against `"draft": false` 4 of 4 equal, four image requests sent together (served one after another here) equal to alone, a text request beside them equal. `text-after-image.txt`: a 200-token text reply before and after an image request has the same sha and speed (73.4 then 72.4 tok/s, 106 rounds), so the CUDA graphs come back after the eager image rounds; the image reply itself ran at 99 tok/s (its answer repeats the question's words, copy drafts).

`visioncheck-final.txt`, `pytest-final.txt`: the same probes (all three OK) and upstream's full suite on the final build (127 failed, 4114 passed, 175 skipped, 11 errors; pristine 127 / 4090 / 175 / 11, identical failing set) (the request-message field order changed after the A/B so upstream's `test_flashnext_message_points.py` stays green; no GPU code changed).

`frozen-c1-final.out`: `bench_decode.py --concurrency 1` on the final build with `VISION=1`, a later request on the same boot: prose 71.1 tok/s, structured 242.4, TTFT 58-59 ms after the first request of each phase (3-run medians).
