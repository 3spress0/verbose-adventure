"""
tgpt_backend.llm_client
========================

A drop-in replacement for the OpenAI client used by the *Generative Agents*
simulation, but driven by the free, no-API-key ``tgpt`` CLI
(https://github.com/aandrew-me/tgpt) and/or the ``python-tgpt`` package
(https://github.com/Simatwa/python-tgpt).

Why?
----
The original ``generative_agents`` codebase (Park et al., 2023) hard-codes
calls to ``openai.ChatCompletion.create`` / ``openai.Completion.create`` /
``openai.Embedding.create``, which requires a paid OpenAI API key. This
module exposes a tiny ``LLMClient`` with the same surface the simulation
expects (``chat(prompt, model=...)``, ``complete(prompt, ...)`` and
``embed(text)``) so we can swap providers without touching any of the
cognitive-modules code.

It supports three backends, picked automatically at runtime:

1. **tgpt** (Go CLI, preferred) — shells out to the ``tgpt`` binary
   installed on the system. Many free providers ship with tgpt
   (``phind``, ``koboldai``, ``pollinations``, ``deepseek``,
   ``groq``, ``ollama`` if you run a local one, etc.). See
   https://github.com/aandrew-me/tgpt for the full list. The exact
   provider can be chosen via the ``TGPT_PROVIDER`` env var.

2. **pytgpt** (Python package) — used as a fallback when the ``tgpt``
   binary is not on the PATH. Same provider set, just an in-process
   import.  See https://github.com/Simatwa/python-tgpt.

3. **Echo** — last-resort offline stub that returns a deterministic
   placeholder. The simulation still *runs* (and you can see agents
   moving on the map) but no real language model is involved. Useful
   for smoke-testing the pipeline without network access.

Configuration is via environment variables (see :func:`_config`) or a
``config`` dict passed to :class:`LLMClient`.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Runtime configuration for the LLM client.

    Environment-variable defaults (so the simulation "just works"):

    - ``TGPT_PROVIDER``        e.g. ``phind``, ``koboldai``,
      ``pollinations``, ``deepseek``, ``auto``. Default: ``auto`` (tgpt
      picks a working one if available).
    - ``TGPT_MODEL``           Model hint forwarded to the backend
      (provider-specific). Empty by default.
    - ``TGPT_BINARY``          Path to the ``tgpt`` binary, default
      ``tgpt`` (resolved via :func:`shutil.which`).
    - ``TGPT_TIMEOUT``         Per-call timeout in seconds, default 90.
    - ``TGPT_BACKEND``         Force a backend: ``tgpt`` / ``pytgpt`` /
      ``echo``. Default: auto-detect.
    - ``TGPT_TEMPERATURE``     Forwarded as ``--temperature`` when set.
    - ``TGPT_EXTRA_ARGS``      Extra CLI args passed to ``tgpt``
      (whitespace-separated). Default empty.
    """

    provider: str = "auto"
    model: str = ""
    tgpt_binary: str = "tgpt"
    timeout: int = 90
    temperature: Optional[float] = None
    extra_args: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        cfg = cls(
            provider=os.environ.get("TGPT_PROVIDER", "auto"),
            model=os.environ.get("TGPT_MODEL", ""),
            tgpt_binary=os.environ.get("TGPT_BINARY", "tgpt"),
            timeout=int(os.environ.get("TGPT_TIMEOUT", "90")),
            temperature=_maybe_float(os.environ.get("TGPT_TEMPERATURE")),
            extra_args=os.environ.get("TGPT_EXTRA_ARGS", "").split(),
        )
        return cfg


def _maybe_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Backend capability detection
# ---------------------------------------------------------------------------

def _have_tgpt_binary(binary: str = "tgpt") -> Optional[str]:
    """Return absolute path to the ``tgpt`` binary, or ``None`` if missing."""
    return shutil.which(binary)


def _have_pytgpt() -> bool:
    """True iff the ``pytgpt`` Python package can be imported."""
    try:
        import pytgpt  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class LLMClient:
    """Drop-in replacement for the OpenAI client surface used by
    ``generative_agents``.

    The class auto-selects the best available backend at construction:

    1. ``tgpt`` binary on the PATH -> call it as a subprocess.
    2. ``pytgpt`` Python package -> import and call in-process.
    3. ``echo`` stub -> return a deterministic placeholder.

    All real backends are configured to NOT need an OpenAI API key.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()
        self.backend = self._select_backend()
        self._pytgpt_provider = None
        if self.backend == "pytgpt":
            self._init_pytgpt()

    # -- public surface -----------------------------------------------------

    def chat(self, prompt: str, model: Optional[str] = None) -> str:
        """Return a chat-style response to ``prompt``.

        ``model`` is accepted for API compatibility with the original
        ``openai.ChatCompletion.create`` interface. We forward it to the
        backend when set.
        """
        return self._dispatch(prompt, model=model)

    def complete(self, prompt: str, **gpt_parameter: Any) -> str:
        """Return a completion-style response to ``prompt``.

        ``gpt_parameter`` may include ``engine``, ``max_tokens``,
        ``temperature``, ``top_p``, ``frequency_penalty``,
        ``presence_penalty``, ``stream``, ``stop``. We forward only
        ``temperature`` (the rest are silently ignored, since most free
        providers don't expose them).
        """
        model = gpt_parameter.get("engine") or gpt_parameter.get("model")
        return self._dispatch(prompt, model=model)

    def embed(self, text: str) -> List[float]:
        """Return a numeric vector for ``text``.

        Neither ``tgpt`` nor ``pytgpt`` exposes a free embedding endpoint
        today, so we use a deterministic, hash-based pseudo-embedding.
        It's not semantically meaningful, but it's stable, fast, and
        good enough for a smoke-test run of the simulation.  Replace
        with a real embedding model (e.g. ``sentence-transformers``)
        for production use — see :mod:`tgpt_backend.embeddings`.
        """
        from .embeddings import hash_embedding  # local import -> avoids
        return hash_embedding(text)

    # -- backend plumbing ---------------------------------------------------

    def _select_backend(self) -> str:
        forced = os.environ.get("TGPT_BACKEND", "").lower().strip()
        if forced in ("tgpt", "pytgpt", "echo"):
            if forced == "tgpt" and not _have_tgpt_binary(self.config.tgpt_binary):
                print(
                    f"[tgpt_backend] TGPT_BACKEND=tgpt but the "
                    f"'{self.config.tgpt_binary}' binary is not on PATH; "
                    "falling back.",
                    file=sys.stderr,
                )
            elif forced == "pytgpt" and not _have_pytgpt():
                print(
                    "[tgpt_backend] TGPT_BACKEND=pytgpt but the "
                    "'pytgpt' package is not installed; falling back.",
                    file=sys.stderr,
                )
            else:
                return forced

        if _have_tgpt_binary(self.config.tgpt_binary):
            return "tgpt"
        if _have_pytgpt():
            return "pytgpt"
        return "echo"

    def _init_pytgpt(self) -> None:
        # pytgpt exposes many provider classes. We pick the same one the
        # user would have chosen for the tgpt binary, defaulting to
        # phind (no API key, free, no signup).
        from pytgpt import phind  # type: ignore

        provider_name = self.config.provider
        provider_mod = None
        if provider_name and provider_name not in ("auto", "phind"):
            try:
                provider_mod = __import__(f"pytgpt.{provider_name}",
                                          fromlist=["*"])
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[tgpt_backend] Could not load pytgpt.{provider_name} "
                    f"({exc}); falling back to phind.",
                    file=sys.stderr,
                )
                provider_mod = None
        if provider_mod is None:
            provider_mod = phind
        cls_name = next(
            (n for n in dir(provider_mod)
             if n.lower().startswith(provider_name or "phind")),
            "PHIND",
        )
        provider_cls = getattr(provider_mod, cls_name)
        try:
            self._pytgpt_provider = provider_cls(is_conversation=False)
        except TypeError:
            # Some pytgpt providers require different kwargs.
            self._pytgpt_provider = provider_cls()

    def _dispatch(self, prompt: str, model: Optional[str] = None) -> str:
        if self.backend == "tgpt":
            return self._call_tgpt(prompt, model=model)
        if self.backend == "pytgpt":
            return self._call_pytgpt(prompt, model=model)
        return self._call_echo(prompt)

    # -- tgpt (Go) backend --------------------------------------------------

    def _call_tgpt(self, prompt: str, model: Optional[str] = None) -> str:
        cmd = [
            self.config.tgpt_binary,
            "--whole",  # return the full reply at once (no streaming)
            "--quiet",  # no spinner
        ]
        if self.config.provider and self.config.provider != "auto":
            cmd += ["--provider", self.config.provider]
        chosen_model = model or self.config.model
        if chosen_model:
            cmd += ["--model", chosen_model]
        if self.config.temperature is not None:
            cmd += ["--temperature", str(self.config.temperature)]
        cmd += self.config.extra_args
        # The prompt is the trailing positional argument.
        cmd += [prompt]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                check=False,
            )
        except FileNotFoundError:
            print("[tgpt_backend] tgpt binary disappeared at runtime; "
                  "falling back to echo.", file=sys.stderr)
            return self._call_echo(prompt)
        except subprocess.TimeoutExpired:
            print(f"[tgpt_backend] tgpt call timed out after "
                  f"{self.config.timeout}s; returning echo fallback.",
                  file=sys.stderr)
            return self._call_echo(prompt)

        if result.returncode != 0:
            print(
                f"[tgpt_backend] tgpt exited {result.returncode}: "
                f"{result.stderr.strip()[:300]}",
                file=sys.stderr,
            )
            return self._call_echo(prompt)

        return result.stdout.strip()

    # -- pytgpt (Python) backend -------------------------------------------

    def _call_pytgpt(self, prompt: str, model: Optional[str] = None) -> str:
        try:
            return self._pytgpt_provider.chat(prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"[tgpt_backend] pytgpt call failed ({exc!r}); "
                  f"returning echo fallback.", file=sys.stderr)
            return self._call_echo(prompt)

    # -- echo backend (smoke-test) -----------------------------------------

    def _call_echo(self, prompt: str) -> str:
        """Deterministic placeholder response.

        The simulation parses responses as JSON. We hand back a
        syntactically valid blob so the pipeline runs to completion even
        without any real LLM.
        """
        # If the prompt looks like it wants JSON output (the
        # ``ChatGPT_safe_generate_response`` wrapper asks for
        # ``{"output": "..."}``), return that shape.
        if '"output"' in prompt and 'json' in prompt.lower():
            return '{"output": "I am unable to respond at the moment."}'
        # Conversation prompts want a list of [name, utterance] pairs.
        if 'Utterance' in prompt or 'conversation' in prompt.lower():
            return (
                '{"output": [["Alex", "I cannot speak right now."], '
                '["Sam", "I understand."]]}'
            )
        return "tgpt_backend echo: no real LLM backend is configured."


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """Return a process-wide singleton :class:`LLMClient`."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = LLMClient()
    print(f"[tgpt_backend] backend = {client.backend}", file=sys.stderr)
    print(client.chat("In one short sentence, what is a generative agent?"))
