#!/usr/bin/env python3
"""Greedy replies with and without drafts (`"draft": false`) from one or more servers: sha of the text, tokens, tok/s.
usage: cmp_drafts.py URL [URL2] [--tokens N] [--dump texts.json]"""
import hashlib, json, sys, time, urllib.request

PROMPTS = [
    "Explain, in about 150 words, why the sky is blue and sunsets are red.",
    "Write a Python function that parses an ISO-8601 date string and returns the day of the week, with a docstring and two examples.",
    "List five differences between TCP and UDP as a numbered list, one sentence each.",
    "Translate to French: 'The committee postponed the vote until the engineers had finished their report.'",
    "Write a bash script that renames every .jpeg file in a directory tree to .jpg, skipping files whose target name exists, and prints a summary.",
    "A train leaves at 09:40 and arrives at 14:05 after a 25-minute stop. What is its moving time? Show the steps briefly.",
]

def ask(url, prompt, tokens, draft):
    body = {"model": "TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP", "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": False}
    if draft is False:
        body["draft"] = False
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.time() - t0
    text = d["choices"][0]["message"].get("content") or ""
    n = d.get("usage", {}).get("completion_tokens", 0)
    return hashlib.sha256(text.encode()).hexdigest()[:12], n, n / dt if dt else 0, text

argv = sys.argv[1:]
tokens = 256
if "--tokens" in argv:
    i = argv.index("--tokens")
    tokens = int(argv[i + 1])
    del argv[i:i + 2]
dump = None
if "--dump" in argv:
    i = argv.index("--dump"); dump = argv[i + 1]; del argv[i:i + 2]
urls = argv
dumped = {}
for i, p in enumerate(PROMPTS):
    row = []
    texts = {}
    for u in urls:
        for draft in (True, False):
            sha, n, tps, text = ask(u, p, tokens, draft)
            row.append(f"{u.split('//')[1].split('/')[0]}{'' if draft else ' nodraft'}: {sha} {n}tok {tps:.1f}tok/s")
            texts[(u, draft)] = text
            dumped[f"{i}:{'draft' if draft else 'nodraft'}"] = text
    print(f"[{i}] " + " | ".join(row), flush=True)
if dump:
    json.dump(dumped, open(dump, "w"), indent=1)
