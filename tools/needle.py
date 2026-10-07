#!/usr/bin/env python3
"""Needle in a long document: usage: needle.py URL TOKENS [DEPTH_FRACTION] [MODEL]. Prints prompt tokens, time to reply,
prefill rate and whether the secret came back. FOLLOWUP=1 adds a 200-token question on the same document (decode behind that context)."""
import json, os, random, sys, time, urllib.request

url, target = sys.argv[1], int(sys.argv[2])
depth = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
model = sys.argv[4] if len(sys.argv) > 4 else "TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP"
rng = random.Random(target + int(depth * 1000))
TOPICS = ["harbour logistics", "glacier surveys", "orchard irrigation", "medieval bookbinding", "railway signalling",
          "coral reef acoustics", "bread fermentation", "tunnel ventilation", "lighthouse optics", "beekeeping records"]
VERBS = ["measured", "reported", "revised", "catalogued", "compared", "scheduled", "inspected", "archived"]
def para(i):
    t = TOPICS[i % len(TOPICS)]
    return (f"Section {i}. The {t} team {rng.choice(VERBS)} {rng.randint(3, 900)} items during week {rng.randint(1, 52)}, "
            f"noting that {rng.choice(TOPICS)} depends on {rng.choice(TOPICS)} more than expected. "
            f"Their log lists {rng.randint(10, 99)} entries, the longest about {rng.choice(TOPICS)}, and ends with a reminder "
            f"to compare results with the {rng.randint(1990, 2025)} baseline before the next review.\n\n")
chars_per_token = 3.9        # calibrated on a first run; the reply's usage gives the real count
needle = "The secret code for the blue door is 7-4-1-9-2. Remember it.\n\n"
n = int(target * chars_per_token / 300) + 1
paras = [para(i) for i in range(n)]
paras.insert(int(len(paras) * depth), needle)
doc = "".join(paras)
q = "\n\nQuestion: What is the secret code for the blue door? Answer with the digits only."
body = {"model": model, "messages": [{"role": "user", "content": "Read this document carefully.\n\n" + doc + q}],
        "max_tokens": 24, "temperature": 0}
t0 = time.time()
req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    d = json.load(r)
dt = time.time() - t0
text = d["choices"][0]["message"].get("content") or ""
pt = d.get("usage", {}).get("prompt_tokens")
ok = "74192" in text.replace("-", "").replace(" ", "")
print(f"prompt_tokens={pt} depth={depth} time={dt:.0f}s ({(pt or 0)/dt:.0f} tok/s prefill) found={ok} reply={text.strip()[:80]!r}", flush=True)
if os.environ.get("FOLLOWUP"):      # decode speed at this depth: the same document (prefix cached), a 200-token answer
    q2 = "\n\nQuestion: Summarize what the document's teams measured, in about 150 words."
    body["messages"][0]["content"] = "Read this document carefully.\n\n" + doc + q2
    body["max_tokens"] = 200
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200) as r:
        d = json.load(r)
    dt = time.time() - t0
    u = d.get("usage", {}); ct = u.get("completion_tokens", 0); cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"followup: cached_tokens={cached} completion_tokens={ct} time={dt:.1f}s ({ct/dt:.1f} tok/s incl. ttft)", flush=True)
