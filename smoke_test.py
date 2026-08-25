#!/usr/bin/env python3
"""
Top-level smoke test for the combined Generative Agents × tgpt project.

Verifies that:
  1. tgpt_backend can be imported and selects a working backend.
  2. The patched gpt_structure.py in generative_agents works without
     an OpenAI API key.
  3. The embedding shim returns a 1536-dim, L2-normalised vector.
  4. The simulation engine's persona modules can be imported.

Run from the top-level directory:

    python3 smoke_test.py
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def header(msg: str) -> None:
    print("\n=== " + msg + " ===")


def main() -> int:
    sys.path.insert(0, REPO)
    failed = 0

    # 1. tgpt_backend
    header("tgpt_backend")
    try:
        from tgpt_backend import LLMClient, get_default_client
        client = get_default_client()
        print(f"  backend = {client.backend}")
        print(f"  chat() works: {bool(client.chat('hi'))}")
        emb = client.embed("smoke test")
        assert len(emb) == 1536, f"expected 1536 dims, got {len(emb)}"
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-6, f"vector is not unit-normalised: {norm}"
        print(f"  embed() works: dim=1536, norm={norm:.6f}")
    except Exception as exc:
        print(f"  FAILED: {exc!r}")
        failed += 1

    # 2. Patched gpt_structure.py
    header("generative_agents.patched_gpt_structure")
    try:
        ga_backend = os.path.join(REPO, "generative_agents", "reverie", "backend_server")
        sys.path.insert(0, ga_backend)
        from persona.prompt_template.gpt_structure import (
            ChatGPT_request,
            ChatGPT_safe_generate_response,
            GPT_request,
            get_embedding,
        )
        print(f"  ChatGPT_request works: {bool(ChatGPT_request('hi'))}")

        def _validate(x, prompt=""):
            return isinstance(x, str) and len(x) > 0
        out = ChatGPT_safe_generate_response(
            "say hi",
            "hi",
            "Output should be a short greeting.",
            repeat=1,
            func_validate=_validate,
            func_clean_up=lambda x, prompt="": x,
        )
        print(f"  ChatGPT_safe_generate_response -> {out!r}")
        assert out, "expected non-empty output"

        v = get_embedding("hello world")
        assert len(v) == 1536
        print(f"  get_embedding works: dim=1536")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {exc!r}")
        failed += 1

    # 3. Persona modules importable
    header("generative_agents.persona_imports")
    try:
        from persona.cognitive_modules import (  # noqa: F401
            perceive, retrieve, reflect, plan, converse, execute,
        )
        print("  cognitive_modules imported")
        import persona.persona  # noqa: F401
        print("  persona.persona imported")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {exc!r}")
        failed += 1

    # Summary
    header("Summary")
    if failed:
        print(f"  {failed} check(s) FAILED")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
