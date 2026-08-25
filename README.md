# Generative Agents × tgpt × python-tgpt

This repository combines three projects into one integrated, **API-key-free**
smallville-style generative-agent simulation:

| Path | Upstream | What it is |
| --- | --- | --- |
| `generative_agents/` | [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents) | The original *Generative Agents* codebase from Park et al. (2023) — a Smallville simulator where 25+ LLM-driven agents live, plan, talk, and remember. |
| `tgpt/` | [aandrew-me/tgpt](https://github.com/aandrew-me/tgpt) | A Go CLI for chatting with free, no-API-key LLM providers (`phind`, `koboldai`, `pollinations`, `deepseek`, `groq`, `ollama`, etc.). |
| `python-tgpt/` | [Simatwa/python-tgpt](https://github.com/Simatwa/python-tgpt) | A Python wrapper for the same providers, plus a `pytgpt` package you can import directly. |
| `tgpt_backend/` | _this repo_ | A thin shim that replaces the OpenAI client in `generative_agents` with calls to `tgpt` / `pytgpt`. **No API key required.** |

The point of the merge is simple: the upstream `generative_agents` repo
hard-codes the OpenAI API; this combined repo swaps that for the free,
no-signup `tgpt` stack so anyone can run a simulation with zero paid
accounts.

---

## TL;DR

```bash
# 1. Install the Go CLI (any one of these works)
go install github.com/aandrew-me/tgpt/v2@latest         # build from source
# or grab a release binary from https://github.com/aandrew-me/tgpt/releases

# 2. Pick a free provider
export TGPT_PROVIDER=phind            # or: pollinations, koboldai, deepseek, ollama, auto, ...

# 3. Install the simulation's Python deps
cd generative_agents
pip install -r requirements.txt
pip install -r reverie/backend_server/requirements.txt   # (already in the top-level file)

# 4. Run the simulation (see "Running the simulation" below)
python reverie/backend_server/reverie.py
```

If the `tgpt` binary is not on your `PATH`, install the `python-tgpt`
package — the same providers are auto-discovered:

```bash
pip install -e python-tgpt
```

The backend is selected at runtime: `tgpt` binary → `pytgpt` Python package
→ offline `echo` stub (smoke test). See `tgpt_backend/llm_client.py` for
the dispatch logic.

---

## What changed vs. the upstream `generative_agents` repo?

Just three files:

| File | Change |
| --- | --- |
| `generative_agents/reverie/backend_server/utils.py` | Replaced the user-supplied OpenAI key with a config block that reads `TGPT_PROVIDER`, `TGPT_MODEL`, etc. from the environment. **No API key needed.** |
| `generative_agents/reverie/backend_server/persona/prompt_template/gpt_structure.py` | Removed `import openai`; every LLM call now goes through `tgpt_backend.llm_client.LLMClient`. Public function signatures (`ChatGPT_request`, `GPT_request`, `safe_generate_response`, `get_embedding`, …) are unchanged. |
| `generative_agents/reverie/backend_server/test.py` | Same swap. |
| `tgpt_backend/` (new) | The shim. It is added to `sys.path` automatically by the new `utils.py`. |

Everything else (the `environment/` Django frontend, the `reverie/`
backend, the `persona/cognitive_modules/*` files, the simulation
storage, the visual map) is untouched, full-fidelity, copied from the
upstream repo. The combined repo carries **all 210k+ files from the
upstream `generative_agents` repo** verbatim, including the precomputed
`base_the_ville_isabella_maria_klaus` simulation storage and the
`July1_the_ville_isabella_maria_klaus-step-3-20` replay.

If you ever want to go back to the OpenAI-backed version, see the
"Restoring the OpenAI backend" section below.

---

## How the integration works

The original `gpt_structure.py` looked like this:

```python
import openai
openai.api_key = openai_api_key
...
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)
return completion["choices"][0]["message"]["content"]
```

The patched version routes every call through `tgpt_backend.LLMClient`,
which auto-selects one of three backends:

1. **`tgpt` (Go CLI)** — shells out to the `tgpt` binary:
   ```
   tgpt --whole --quiet --provider phind "Hello there"
   ```
2. **`pytgpt` (Python package)** — in-process, same providers.
3. **`echo`** — deterministic stub that returns valid JSON of the shape
   the simulation expects, so the pipeline runs end-to-end with no LLM
   (useful for smoke-testing the rendering / movement loop).

Configuration is via env vars (defaults shown):

| Env var | Default | Meaning |
| --- | --- | --- |
| `TGPT_PROVIDER` | `auto` | Provider name (`phind`, `koboldai`, `pollinations`, `deepseek`, `groq`, `ollama`, `auto`, …) |
| `TGPT_MODEL` | `""` | Model hint forwarded as `--model` |
| `TGPT_BINARY` | `tgpt` | Path to the `tgpt` Go binary |
| `TGPT_TIMEOUT` | `90` | Per-call timeout in seconds |
| `TGPT_TEMPERATURE` | `""` | Sampling temperature (provider-specific) |
| `TGPT_EXTRA_ARGS` | `""` | Whitespace-separated extra CLI args |
| `TGPT_BACKEND` | _auto_ | Force `tgpt` / `pytgpt` / `echo` (skip auto-detect) |

### Embeddings

`tgpt` / `pytgpt` do not currently expose a free embedding endpoint, so
`get_embedding` returns a **deterministic hash-based pseudo-embedding**
(1536-dim, L2-normalised). The simulation still runs end-to-end; memory
retrieval will be bag-of-words-ish rather than semantically meaningful.

If you want a real embedding, plug `sentence-transformers` (or any
custom service) into `tgpt_backend/embeddings.real_embedding`. The
function signature is `(text: str) -> List[float]`. The retrieval code
in `persona/cognitive_modules/retrieve.py` then works against your
embedding without any further changes.

---

## Running the simulation

This is the same procedure as the upstream README — we just don't need
the API-key step anymore.

```bash
# 0. Clone this combined repo and cd in
cd combined/

# 1. (Optional) Install tgpt CLI: pick the no-key provider you want.
go install github.com/aandrew-me/tgpt/v2@latest
export TGPT_PROVIDER=phind

# 2. Install generative_agents' Python deps
pip install -r generative_agents/environment/frontend_server/requirements.txt
# (The top-level `requirements.txt` is identical.)

# 3. Start the environment (Django) server
cd generative_agents/environment/frontend_server
python manage.py runserver
# Open http://localhost:8000/ and verify the "environment server is up" message.
# Leave this terminal running.

# 4. Start the simulation (backend) server
cd ../reverie/backend_server
python reverie.py
# When prompted:
#   Enter the name of the forked simulation: base_the_ville_isabella_maria_klaus
#   Enter the name of the new simulation:    test-simulation
# At the "Enter option:" prompt, type:
#   run 100        # simulate 100 game steps (each step = 10 s of sim time)
```

To replay the pre-computed simulation (no LLM calls, just watch it run):

```
http://localhost:8000/replay/July1_the_ville_isabella_maria_klaus-step-3-20/1/
```

To demo a simulation with proper character sprites, run the
`compress` function in `reverie/compress_sim_storage.py` and then visit:

```
http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/
```

### Trying a different LLM provider

```bash
# Use a hosted, no-signup free provider:
export TGPT_PROVIDER=phind
# Or a different one:
export TGPT_PROVIDER=pollinations
# Or a local Ollama server (you must have one running on localhost:11434):
export TGPT_PROVIDER=ollama
export TGPT_MODEL=llama3

python reverie/backend_server/reverie.py
```

### Smoke-testing without any network

Set `TGPT_BACKEND=echo` and the simulation will run with a deterministic
stub. The agents won't produce meaningful dialogue, but you can confirm
the whole rendering / movement / save / load pipeline works.

```bash
TGPT_BACKEND=echo python reverie/backend_server/reverie.py
```

---

## Restoring the OpenAI backend

If you want to use the original OpenAI client (you have an API key, you
prefer GPT-4, etc.):

1. Re-install the upstream `openai` package and restore the original
   `gpt_structure.py`:

   ```bash
   pip install openai==0.28
   git checkout -- generative_agents/reverie/backend_server/
   ```

   _(Note: the upstream code uses the legacy `openai.ChatCompletion`
   API which requires `openai<1.0`.)_

2. Create `reverie/backend_server/utils.py` with your API key as the
   original README describes.

3. Run as usual.

---

## Repository layout

```
combined/
├── README.md                           ← you are here
├── generative_agents/                  ← full upstream repo, full fidelity
│   ├── environment/
│   │   ├── frontend_server/            ← Django env server (the Smallville map)
│   │   └── ...
│   ├── reverie/
│   │   ├── backend_server/             ← the simulation engine
│   │   │   ├── persona/
│   │   │   │   ├── cognitive_modules/  ← perceive, retrieve, plan, reflect, …
│   │   │   │   ├── prompt_template/    ← (patched) gpt_structure.py lives here
│   │   │   │   └── …
│   │   │   ├── utils.py                ← (patched) replaced API-key block
│   │   │   └── …
│   │   ├── compress_sim_storage.py
│   │   └── …
│   ├── requirements.txt
│   └── …
├── tgpt/                               ← full upstream tgpt (Go) repo
│   ├── main.go
│   ├── src/providers/                  ← one subdir per provider
│   ├── go.mod
│   └── …
├── python-tgpt/                        ← full upstream python-tgpt repo
│   ├── src/pytgpt/
│   │   ├── phind/                      ← default free provider
│   │   ├── openai/
│   │   ├── groq/
│   │   ├── gpt4free/
│   │   ├── …
│   ├── setup.py
│   └── …
└── tgpt_backend/                       ← NEW: the API-key-free shim
    ├── __init__.py
    ├── llm_client.py                   ← LLMClient (tgpt / pytgpt / echo dispatch)
    ├── embeddings.py                   ← hash-based pseudo-embedding (1536-dim)
    └── README.md                       ← per-module docs
```

---

## Why this combination works

* **`generative_agents` was built on top of OpenAI**, but the
  simulation only needs an LLM that can follow fairly structured
  prompts and return JSON. Every free provider behind `tgpt` can do
  that.
* **`tgpt`** is a small, well-maintained Go CLI that already supports
  ~20 free providers, has good error messages, and exposes just the
  flags we need (`--provider`, `--model`, `--whole`, `--quiet`).
* **`python-tgpt`** is the same provider system in Python form, so the
  backend can stay in-process when the Go binary isn't available.
* **`tgpt_backend`** is the thinnest possible shim: 300 lines of Python
  that replaces the OpenAI client surface and adds an
  embedding fallback. It does not fork the simulation code — every
  cognitive module keeps using `ChatGPT_request`, `GPT_request`,
  `safe_generate_response`, and `get_embedding` exactly as before.

---

## License

* `generative_agents/` — Apache 2.0 (see its `LICENSE` file).
* `tgpt/` — see its `LICENSE` file.
* `python-tgpt/` — see its `LICENSE` file.
* `tgpt_backend/` — MIT (this repo).

---

## Credits

Generative Agents was published by Joon Sung Park, Joseph C. O'Brien,
Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S.
Bernstein in 2023 ([arXiv:2304.03442](https://arxiv.org/abs/2304.03442)).
The `tgpt` CLI is maintained by [aandrew-me](https://github.com/aandrew-me);
the `python-tgpt` package is maintained by
[Simatwa](https://github.com/Simatwa). The `tgpt_backend` shim and the
combination of the three is by this repo's author.
