"""
tgpt_backend — API-key-free LLM backend for the Generative Agents simulation.

This package is the glue that lets the *Generative Agents* codebase
(https://github.com/joonspk-research/generative_agents) run without an
OpenAI API key.  It does this by replacing the OpenAI client in
``reverie/backend_server/persona/prompt_template/gpt_structure.py``
with calls to the free ``tgpt`` (Go) CLI and/or the ``pytgpt`` (Python)
package.

Public surface:

- :class:`tgpt_backend.llm_client.LLMClient`
- :class:`tgpt_backend.llm_client.LLMConfig`
- :func:`tgpt_backend.llm_client.get_default_client`
- :func:`tgpt_backend.embeddings.get_embedding`
- :func:`tgpt_backend.embeddings.hash_embedding`
"""
from .llm_client import LLMClient, LLMConfig, get_default_client
from .embeddings import get_embedding, hash_embedding

__all__ = [
    "LLMClient",
    "LLMConfig",
    "get_default_client",
    "get_embedding",
    "hash_embedding",
]

__version__ = "0.1.0"
