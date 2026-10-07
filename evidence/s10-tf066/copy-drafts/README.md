# s10 · copy drafts (docker/patches/flashnext-tools-0.6.6.patch, part 3)

`startup.txt`: both ranks announce "or up to 15 copy drafts where the reply repeats earlier text".
`copy-cells.txt`: five one-stream cells, each run with drafts on (copy + MTP) and again with `"draft": false`; the reply
token shas are equal in every cell. Copying 130 -> 345 tok/s, fixing typos 124 -> 323, a rename refactor 135 -> 288;
fresh prose 72 -> 71 and fresh code 127 -> 130 (no earlier copy, no change). The "before" numbers are
`../copy-vs-fresh.txt`, measured on the previous image the same night.
`bench-concurrent-512.out/json`: TensorFold's ruler at 1 and 4 users with `--alone --serial`: every concurrent reply
equals its solo run and the solo run equals one-token decoding. `frozen-c1.out`: prose 67.9, structured 235.7.
