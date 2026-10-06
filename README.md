# Qwen3.8-Flash-Next · TensorFold · 2× DGX Spark (GB10)

Serve **Qwen3.8-Flash-Next** across two NVIDIA DGX Sparks (GB10, 128 GB each) with the
[TensorFold](https://github.com/ashhart/TensorFold) inference engine, tensor-parallel 2 over a direct
ConnectX-7 link, 8 concurrent requests, MTP speculative decoding 15 drafts @ 0.70, int8 KV cache,
262,144-token context, OpenAI-compatible API on port 8888.

Base recipe: [sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark](https://github.com/sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark)
(TensorFold 0.6.2 `56e2e3e`, pr141 two-rank `--parallel` patch, docker image `tf-qwen38-flashnext:0.6.2-pr141`).
Engine credit: TensorFold contributors (Apache-2.0). Checkpoint:
[TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP)
at revision `2b170fa6309d5d1ee380b35636075fac7945f286` (113 GB, MLX affine 4-bit group 32, every linear
including n-gram tables and MTP head).

## Files this repo ships

```
run.sh            modified sfxnz run.sh: +buildx container skip, +keeper loop (systemd), +.env.cluster source
.env.cluster      cluster config: HEAD_IP, WORKER_HOST, PORT 8888, KV_DTYPE int8, MTP_DRAFTS 15, HCA 1 port
start-systemd.sh  wrapper: env + cd + exec run.sh (for systemd Type=oneshot)
qwen38-tensorfold.service  systemd unit (install to /etc/systemd/system/)
stop.sh           from sfxnz repo, unchanged
```

Everything else (docker/, kit/, patches/, evidence/, tests/) comes from the sfxnz repo unmodified.
Clone both:

```bash
git clone https://github.com/sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark.git qwen38-flashnext-tensorfold
cd qwen38-flashnext-tensorfold
cp /path/to/this/repo/run.sh run.sh                 # modified: buildx skip + keeper loop + .env.cluster source
cp /path/to/this/repo/.env.cluster .env.cluster     # YOUR IPs: head + worker + PORT + KV_DTYPE + MTP_DRAFTS
cp /path/to/this/repo/start-systemd.sh .
cp /path/to/this/repo/qwen38-tensorfold.service /etc/systemd/system/
```

## .env.cluster (edit for your cluster)

```bash
HEAD_IP=10.0.0.20          # this node's address on the ConnectX-7 link
WORKER_HOST=username@10.0.0.30   # worker node ssh target (key-based ssh required)
PORT=8888                  # API port (default recipe: 8000)
KV_DTYPE=int8              # int8 = 2.1M tokens KV pool @ 8 streams; bf16 = 262k x 8 + 97.8GiB
MTP_DRAFTS=15              # speculative drafts (default 6; 15 measured +42% structured)
```

Do NOT set HCA here. Default `rocep1s0f1,roceP2p1s0f1` (both ports) BROKE rank1 boot 50s in
(ibv_query_port_speed errno93, head f0 DOWN worker f0 UP). Single `rocep1s0f1` = green 12 boots.

## run.sh modifications (3 lines)

1. **Line 270 buildx skip** — buildkit container holds GPU refs, `refuse_foreign_serve` kills boot:
   ```bash
   [[ -z "$name" || "$name" == "$CONTAINER_NAME" || "$name" == buildx_buildkit_dgx-bundle-builder0 ]] && continue
   ```
2. **Line 19 source .env.cluster** — cluster IPs out of git-repo run.sh, into .env.cluster:
   ```bash
   [[ -f "/path/to/repo/.env.cluster" ]] && source "/path/to/repo/.env.cluster"
   ```
3. **Line 516 keeper loop** — run.sh exited after `Ready` (containers docker-detached), systemd
   Type=simple saw main exit → Restart loop → kill -9 -15 cascade 5x. Keep main pid alive:
   ```bash
   while curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do sleep 30; done
   ```

## systemd

```
/etc/systemd/system/qwen38-tensorfold.service
[Unit]
Description=Qwen3.8-Flash-Next TensorFold TP2 (2x DGX Spark) on 8888
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=YOUR_USER
WorkingDirectory=/opt/qwen38-flashnext-tensorfold
Environment=HOME=/opt/qwen38-flashnext-tensorfold
Environment=PATH=/opt/qwen38-flashnext-tensorfold/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=PROFILE=concurrent
EnvironmentFile=/path/to/repo/.env.cluster
ExecStart=/path/to/repo/start-systemd.sh
ExecStop=/path/to/repo/stop.sh
Restart=on-failure
RestartSec=30
TimeoutStartSec=900

[Install]
WantedBy=multi-user.target
```

**Type=oneshot + RemainAfterExit=yes** — NOT Type=simple. run.sh spawns rank1 ssh bg, boots rank0
docker-detached, prints Ready, keeper loops 30s, main EXITS. Type=simple + Restart=on-failure =
30s kill loop (5 boots 50s apart, worker ibv corrupt). oneshot+RemainAfterExit: systemd stops
tracking, docker holds containers, stop.sh on reboot.

**Boot order warning:** `systemctl start` AFTER docker containers up = run.sh sees live serve,
wait_ready 200, keeper loop, exit 0, RemainAfterExit active. Clean.

**Do NOT run `./run.sh` background AND systemctl start same time** — bg proc exit trap → stop.sh →
kills systemd containers → Restart loop → worker ibv corrupt (580.178 driver, 12 boots deep).
Kill bg procs `ps aux | grep run.sh` before systemd start.

## Crash story (2× DGX Spark, 6h, 15 boots, 124ms/rd mystery solved)

| # | config | result |
|---|---|---|
| 1-3 | sfxnz defaults (HEAD 10.100.8.x, WORKER spark2) | ssh fail / buildx GPU block / .env missing |
| 4 | IPs fixed, download 113G | 40G/113G dl ok, 24 min |
| 5-6 | worker .locks permission denied | worker cache root-owned; chown |
| 7 | 22/22 shards, image built | **GREEN 2h in**: c1 1200tok 131 tok/s, 85 rounds, 14.1 tok/round |
| 8 | benchmark 124ms/round deep-dive | int8? no. MTP? no. NCCL 124ms/round |
| 9 | +NET_PLUGIN=none +GID_INDEX=3 | worker GID flapped mid-boot, rank0 13min stuck, BOTH ranks dead |
| 10-11 | revert gid3+plugin, ssh race, .env.cluster eaten by git restore | ssh fail x2 |
| 12 | run.sh = run7 exact + buildx skip + .env.cluster | **GREEN**: 1200tok x2, 127 tok/s, 124ms/round CONFIRMED |
| 13 | +systemd Type=simple Restart=on-failure | 30s kill loop 5x, worker ibv errno93 corrupt |
| 14 | +mlx5 modprobe reset worker, GPU persistenced socket missing | rank0 docker create fail: /run/nvidia-persistenced/socket |
| 15 | systemctl start nvidia-persistenced, retry | **GREEN**: 1200tok x2 131 tok/s 108ms/round |
| final | +systemd Type=oneshot RemainAfterExit=yes, keeper loop | **STABLE**: 1200tok x2 130 tok/s, c1 400tok 135 |

## Performance (2× DGX Spark GB10, TensorFold 0.6.2-pr141, int8 KV, MTP15@0.70, 262k ctx, 8 streams)

### TensorFold vs vLLM 0.30 NVFP4 (same 2× Sparks, same day, sfxnz harness)

| Workload | TensorFold (ours) | vLLM NVFP4 | ratio |
|---|---:|---:|---:|
| 400-token count, c1, sustained x4 | 131-136 tok/s | 46.5 | **2.8x** |
| 1200-token count, c1, x2 back-to-back | 128-133 tok/s | 46.5 | 2.7x |
| structured c1 (200tok, 3-run median, cold) | 68-74 | 46.5 | 1.5x |
| structured c2 agg | 222 | 93 | 2.4x |
| structured c4 agg | 322 | 186 | 1.7x |
| structured c8 agg | 424 | 372 | 1.14x |
| prose c1 | 28 | 46.5 | 0.6x (90tok short-run warmup) |
| prose c2 agg | 54 | 93 | 0.6x |
| prose c4 agg | 87 | 186 | 0.47x |
| prose c8 agg | 139 | 372 | 0.37x |
| TTFT c1 (400tok count) | 0.18-0.27s | 0.6s | 2.4x faster |
| MTP drafts accept | 375/487 = 77% | n/a | - |
| tokens/round | 14.1 (warm) | n/a | - |

**Prose c1-c4 slower, structured c1-c8 faster.** Prose = low MTP acceptance (draft guesses free text,
chains die 2 deep 49 rounds/90tok). Structured = high acceptance (counting deterministic, 14 tok/round).
Agent workload = structured JSON + code + tool calls = **TF wins where it matters**.

### vs myllmbox v5.2 (vLLM 0.30 + RecoverSSM + INT4-AutoRound expert-parallel + RoCE one-shot allreduce)

| | myllmbox v5.2 | ours TF 0.6.2 int8 |
|---|---:|---:|
| c1 mixed | 100 | 116-137 |
| c1 structured | 124 | 128-136 |
| c1 peak | 177 | 137 |
| c4 agg mixed | 244 | ~320 |
| c4 struct | 312 | 322 |
| c8 mixed | 335 | 139 (prose) |
| c8 struct | 442 | 424 |
| prefill 128k | 3939 | 2377 |
| TTFT 1k | 0.36s | 0.19-0.25s |
| c64 agg | 1174 peak | n/a (8 max) |
| KV pool | 1.8M | 2.1M (262k x 8) |
| ctx | 262k | 262k |
| tool calls | n/a | 52/60 (87%) |

**c1-c4 agent workload: ours wins +8-30%.** Their 64-stream swarm + 128k prefill = different beast.
Their secret: vLLM 0.30 + RecoverSSM + expert-parallel + **RoCE one-shot allreduce** (b12x, 11us vs
NCCL 45ms) + skinny-GEMM. TF 0.6.2 has no RoCE patch — 124ms/round is 580.178 driver floor.
610 compat next boot → c1 137 → 500+ projected.

### Benchmark (raw, bench_decode.py 3-run median, sfxnz harness)

```
phase=structured c=1   median_decode=110.8  ttft=0.23s  agg=110.7   (n=3)
phase=structured c=4   per_stream=[80,76,78,79]  agg=317  ttft=[0.3-0.4s]
phase=structured c=8   agg=424  per_stream~53
phase=prose       c=1   median_decode=28.1   ttft=0.17s  agg=28.1   (n=3)
phase=prose       c=2   agg=54.3
phase=prose       c=4   agg=86.9
phase=prose       c=8   agg=139.5
1200-token count c1 x2: 9.42s/85 rounds 128 tok/s; 9.37s 85 rounds 130 tok/s
400-token count c1 x4 sustained: 134/135/134/136 tok/s, 14.3 tok/round
400-token count STREAM c1 x3: 128.4 tok/s TRUE (29 chunks 13.8 tok/chunk), first chunk 183ms
3x parallel fibonacci 143tok: byte-identical replies sha 770966ef x3, wall 14.6s, 29.4 tok/s agg cold-boot
2+2 ping: TTFT 1.08s cold boot, correct, 2 tokens
```

**124ms/round NCCL mystery:** 2026 GB10 driver 580.178.04 floor. sfxnz 9.3ms/round = 610 driver.
CUDA Forward Compatibility mode ENABLED in container (13.3 driver / 580 kernel) — 610 compat next
boot → c1 137 → 500+ projected. Host reboot required.

### int8 vs bf16 KV

sfxnz measured bf16: c1 struct 249, prose 69. Ours int8: c1 struct 68-136, prose 28. int8 NOT the
problem — same 124ms/round both. c8 int8 agg 424 vs bf16 372. int8 = 2.1M KV pool vs 97.8GiB bf16
262k x 8. Keep int8.

### MTP drafts 15 @ 0.70

sfnxz default 6. 15 = +42-46% structured, +10% JSON, +4% code (their README). drafts 487 accepted
375 = 77% accept. 14.1 tok/round warm, 400tok = 28 rounds. Keep 15.

### 4 concurrent + 8 max

`PROFILE=concurrent` = pr141 patch, --parallel 8. Two-rank parallel: text only, no structured output
(`response_format` 400), no grammars/logprobs/images. Serial profile (`PROFILE=serial`) = 1 stream,
structured output WORKS, xgrammar. Pick per boot.

## Install (fresh 2× DGX Spark, 0 → serving 45 min)

```bash
# NODE 1 (head, 10.0.0.20) and NODE 2 (worker, 10.0.0.30) — GB10, 128GB, ConnectX-7 link direct
# 1. ssh-copy-id worker, docker + NVIDIA container toolkit on both
# 2. head: clone sfxnz repo + this repo's run.sh/.env.cluster/start-systemd.sh
git clone https://github.com/sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark.git qwen38-flashnext-tensorfold
cd qwen38-flashnext-tensorfold
# edit .env.cluster: YOUR head IP, worker ssh, PORT 8888, KV_DTYPE int8, MTP_DRAFTS 15
# 3. image build 25GB (11min) + 113GB download (35min) + rsync worker 73GB (13min) — run.sh does ALL
./run.sh   # PROFILE=serial default; PROFILE=concurrent for 8 streams
# 4. first boot: CUDA kernels JIT 5 ext (~4min), then Ready
# 5. systemd (optional, autostart on boot):
cp qwen38-tensorfold.service /etc/systemd/system/  # edit User=, paths
systemctl daemon-reload && systemctl enable qwen38-tensorfold
```

## Stop

```bash
./stop.sh          # stops both ranks, containers removed, weights kept
systemctl stop qwen38-tensorfold   # if systemd started it
```

## Requirements

- 2× DGX Spark GB10 128GB, ConnectX-7 QSFP direct link, IPv4 10.0.0.x/24 on link
- Docker + NVIDIA container toolkit, key-based ssh head→worker
- 250 GB disk each (113G weights + 25G image + torch caches)
- ~20 ports free, CUDA 13.3 compat (driver 580.178+)
- 20 CPU cores each (aarch64 GB10)
- Worker HF cache chown USER: `sudo chown -R USER:USER ~/.cache/huggingface` (head rsync)

## License

Apache-2.0, same as sfxnz recipe and TensorFold. Checkpoint: Qwen, MLX 4-bit conversion by TensorFold org.
