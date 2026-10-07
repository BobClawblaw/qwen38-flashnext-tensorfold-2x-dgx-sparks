# s10 · one user: serial profile (PARALLEL=1) against the concurrent profile (PARALLEL=8), same 0.6.6 image

Four cells per profile: `tools/bench_distinct.py` (12 different prompts, 256 tokens, the realistic one-user case),
`bench_decode.py --concurrency 1` (frozen ruler), count 1 to 400, and `tools/vendor/bench_concurrent.py --levels 1 --tokens 512`.
The concurrent profile was measured three times on this boot: right after the 1M-context session (`*-concurrent.*`), right after
the systemd restart (`*-concurrent-freshboot.*`), and once more a few minutes later (`frozen-c1-concurrent-freshboot-2.out`).

| Cell | concurrent, after 1M-context session | concurrent, fresh restart | concurrent, second pass | serial | serial, booted as default |
|---|---:|---:|---:|---:|---:|
| distinct prompts, median tok/s | 55.3 | 62.1 | | **78.9** | |
| frozen prose c=1 | 54.4 | 56.0 | 65.6 | **71.6** | **73.3** |
| frozen structured c=1 | 190.8 | 199.3 | 239.6 | **241.9** | **243.6** |
| code 512 c=1 | 87.5 | 87.1 | | **110.7** | |
| chat 512 c=1 | 77.3 | 78.8 | | **96.8** | |
| count 1-400 | 165.8 | | | **226.2** | |

The concurrent decoder pays a per-prompt warm-up: a repeated prompt speeds up pass to pass (56.0 then 65.6 on the same boot),
and a different prompt every time stays at 55-62. The serial engine captures 48 CUDA graphs at startup and has no plan or
digest exchange per round; its distinct-prompt rate is 27-43% higher. For one user at a time, serve `PARALLEL=1`.

The last column is a later boot with `PROFILE=serial` set in the unit (the recipe's default profile) rather than a `PARALLEL=1` restart of the concurrent one: `frozen-c1-serial-boot.out`, 3-run medians, the first request of each phase pays a 100-125 ms TTFT and the rest 53-61 ms.
