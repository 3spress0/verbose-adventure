"""
File: utils.py
Description: Configuration / paths for the Generative Agents simulation.

In the original repo this file was created by each user with their
OpenAI API key.  In this combined project we no longer need an OpenAI
key: the LLM is served by the free ``tgpt`` CLI / ``pytgpt`` package
via the :mod:`tgpt_backend` shim.  See the top-level ``README.md`` for
configuration knobs (``TGPT_PROVIDER``, ``TGPT_MODEL``,
``TGPT_BINARY``, ``TGPT_TEMPERATURE``, etc).

If you want to fall back to the original OpenAI behaviour, restore the
``openai_api_key`` block below, install the ``openai`` package and
revert ``persona/prompt_template/gpt_structure.py``.
"""

import os
import sys

# Make sure the tgpt_backend package is importable when the simulation
# is launched from any cwd.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# File-system layout (kept identical to the original repo)
# ---------------------------------------------------------------------------
maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix    = f"{maze_assets_loc}/the_ville/matrix"
env_visuals   = f"{maze_assets_loc}/the_ville/visuals"

fs_storage       = "../../environment/frontend_server/storage"
fs_temp_storage  = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True

# ---------------------------------------------------------------------------
# LLM configuration (consumed by tgpt_backend.llm_client.LLMConfig)
# ---------------------------------------------------------------------------
# To use a real free LLM provider, set TGPT_PROVIDER to one of the
# providers supported by tgpt (e.g. "phind", "koboldai",
# "pollinations", "deepseek", "groq", "ollama", "auto").  See
# https://github.com/aandrew-me/tgpt for the full list.
#
# Examples:
#   export TGPT_PROVIDER=phind
#   export TGPT_PROVIDER=pollinations
#   export TGPT_PROVIDER=ollama            # if you have ollama running
#   export TGPT_MODEL=llama3               # forwarded to --model
TGPT_PROVIDER = os.environ.get("TGPT_PROVIDER", "auto")
TGPT_MODEL    = os.environ.get("TGPT_MODEL", "")

# ---------------------------------------------------------------------------
# (Optional) OpenAI fallback — leave these empty unless you've reverted
# gpt_structure.py to use the OpenAI client.
# ---------------------------------------------------------------------------
# openai_api_key = "<Your OpenAI API>"
# key_owner      = "<Name>"
