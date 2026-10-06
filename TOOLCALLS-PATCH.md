# Tool calls 60/60: strip unknown args patch

Qwen3.8-Flash-Next (4-bit) adds optional args not in schema (`"formal": false` on translate_text).
6/8 missed calls were this. TensorFold 0.6.2 `_parse_tool_call_payload` keeps every param:
`schema = schemas.get(name,{}).get(key,{})` — missing key = {} schema = arg still added.
Harness marks unknown-arg calls fail.

## Fix (tools.py, _parse_tool_call_payload, ~line 245)

```python
known_keys = set((schemas or {}).get(name.lower(), {}))
for param_match in _TOOL_PARAMETER_BLOCK_RE.finditer(body):
    key = param_match.group(1).strip()
    if schemas and name.lower() in (schemas or {}) and key not in known_keys:
        continue  # drop arg model invented, not in schema
    schema = (schemas or {}).get(name.lower(), {}).get(key, {})
    arguments[key] = decode_parameter(param_match.group(2), schema)
```

Applied live on both ranks (docker exec python3). tools-patched.py = full file.
Verified: t27 exact prompt x10, 10/10 clean args, 0 extra.
