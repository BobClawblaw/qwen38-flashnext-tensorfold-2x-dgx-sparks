# s10 · quality gates on 0.6.6 (the patch, copy drafts on, PARALLEL=16), 2026-10-07

`t2/`: `quality/t2.py --tasks gsm8k,ifeval,tools,rep4,effort --workers 4` (manifest, per-item jsonl, summary json, log).
gsm8k 239/250 (95.6%), ifeval 102/120 (85.0%), tools 60/60, rep4 64/64 (mean repeated 4-grams 0.04%), effort 5/5.
`t3/`, `t3-250k/`: `tools/run_t3.sh` needles 4k/16k/64k/128k 24/24 and 250k 6/6 (prompt tokens 246k-250k, 125-128 s cold each).
Datasets: gsm8k test.jsonl (openai/grade-school-math @3101c7d, sha256 3730d312…) and ifeval input_data.jsonl
(google-research @26d8ccd, sha256 67ffeee0…), both checked by the harness against quality/data/t2_ids.json.
`t2-json/`: the json_schema gate (30 prompts, strict) on the serial profile (`PARALLEL=1`): 30/30. The first attempt's serial
start hit an NCCL init race while the previous rank 1 was still tearing down (run-serial.log); the rerun (run-serial-2.log)
waited for both containers to be gone and no process on either GPU before starting.
