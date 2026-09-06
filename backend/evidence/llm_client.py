"""
Pluggable LLM client for the explanation-generation step.

By default, `generator.generate_explanation` is called with no client at
all (`llm_client=None`), which skips the LLM entirely and uses the fully
deterministic template path from templates.py. That's the zero-setup mode
this part ships in, matching the architecture doc's "Mocking strategy: none
needed inward" note for Part 5 — its own tests never need a live model.

`LocalOpenAICompatibleLLMClient` is the real option for the integrated
system: talks to a locally-hosted, OpenAI-chat-compatible server (Ollama,
vLLM, llama.cpp server, etc.) — never a cloud API, per the project's
"local-first, no live external API dependency at request time" rule. Wire
this in from Part 2 once a local model is actually running in the demo
environment; nothing else in this package needs to change.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol


class ExplanationLLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class LLMUnavailableError(RuntimeError):
    """Raised by a client when it cannot produce a completion. generator.py
    always catches this (and any other client error) and falls back to the
    fully-templated sentence — it must never become a user-facing failure.
    """


class LocalOpenAICompatibleLLMClient:
    """Calls a local OpenAI-chat-compatible endpoint, e.g. Ollama's
    `http://localhost:11434/v1/chat/completions`. Uses only the standard
    library, so this part has zero extra install cost beyond pydantic.

    Example (from Part 2, once a local model is running):
        from backend.evidence.llm_client import LocalOpenAICompatibleLLMClient
        from backend.evidence import validate_and_respond

        client = LocalOpenAICompatibleLLMClient(model="llama3.1")
        response = validate_and_respond(query, evidence, llm_client=client)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.1",
        timeout_s: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError) as exc:
            raise LLMUnavailableError(f"local LLM call failed: {exc}") from exc
