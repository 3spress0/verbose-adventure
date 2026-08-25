# `tgpt_backend`

A small, dependency-free shim that lets the
[Generative Agents](../generative_agents) simulation run without an
OpenAI API key.

## What's in here

| File | Purpose |
| --- | --- |
| `llm_client.py` | `LLMClient` + `LLMConfig`. Auto-selects the `tgpt` Go CLI, the `pytgpt` Python package, or an offline `echo` stub. Exposes `chat(prompt, model=...)`, `complete(prompt, **kwargs)`, and `embed(text)`. |
| `embeddings.py` | Deterministic, hash-based 1536-dim pseudo-embedding (drop-in for OpenAI's `text-embedding-ada-002`). Plug in a real embedding model by overriding `real_embedding`. |
| `__init__.py` | Re-exports the public surface. |

## Quick test

```bash
cd ../
python3 -c "
import sys; sys.path.insert(0, '.')
from tgpt_backend import LLMClient
c = LLMClient()
print('backend =', c.backend)
print(c.chat('Say hi.'))
print('embed dim =', len(c.embed('hello')))
"
```

## Environment variables

| Env var | Default | Meaning |
| --- | --- | --- |
| `TGPT_PROVIDER` | `auto` | Provider name passed to the backend |
| `TGPT_MODEL` | `""` | Model hint |
| `TGPT_BINARY` | `tgpt` | Path to the `tgpt` binary |
| `TGPT_TIMEOUT` | `90` | Per-call timeout in seconds |
| `TGPT_TEMPERATURE` | `""` | Sampling temperature |
| `TGPT_EXTRA_ARGS` | `""` | Extra CLI args |
| `TGPT_BACKEND` | _auto_ | Force `tgpt` / `pytgpt` / `echo` |

See `llm_client.py` for the full dispatch logic.
