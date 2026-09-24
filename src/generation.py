from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class TextGenerator(Protocol):
    """Small interface that keeps the RAG pipeline independent of one model provider."""

    def generate(self, prompt: str) -> str:
        ...


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server or requested model cannot be reached."""


@dataclass
class OllamaGenerator:
    """Generate an answer with a model running locally through Ollama."""

    model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout_seconds: int = 120

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaUnavailableError(
                "Ollama is unavailable. Start it with `ollama serve` and pull the configured model."
            ) from exc

        try:
            return str(body["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise OllamaUnavailableError("Ollama returned an unexpected response.") from exc


@dataclass
class ExtractiveFallbackGenerator:
    """Cost-free fallback that returns retrieved evidence when Ollama is not running."""

    max_characters: int = 700

    def generate(self, prompt: str) -> str:
        marker = "CONTEXT:\n"
        context = prompt.split(marker, maxsplit=1)[-1].split("\n\nANSWER:", maxsplit=1)[0]
        first_source = context.split("\n\n[", maxsplit=1)[0]
        evidence = first_source.split("\n", maxsplit=1)[-1].strip()
        if not evidence:
            return "I could not find enough evidence in the indexed documents."
        return f"The strongest retrieved evidence says: {evidence[:self.max_characters]} [1]"


@dataclass
class ResilientLocalGenerator:
    """Prefer Ollama but keep the application usable with no running model server."""

    primary: TextGenerator
    fallback: TextGenerator

    def generate(self, prompt: str) -> str:
        try:
            return self.primary.generate(prompt)
        except OllamaUnavailableError:
            return self.fallback.generate(prompt)
