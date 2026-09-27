from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import httpx
from langchain_ollama import ChatOllama
from ollama import ResponseError


class TextGenerator(Protocol):
    """Small interface that keeps the RAG pipeline independent of one model provider."""

    def generate(self, prompt: str) -> str:
        ...


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server or requested model cannot be reached."""


@dataclass
class OllamaGenerator:
    """Use LangChain's chat-model interface with the existing local Ollama model."""

    model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout_seconds: int = 120

    @cached_property
    def client(self) -> ChatOllama:
        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.1,
            client_kwargs={"timeout": self.timeout_seconds, "trust_env": False},
        )

    def generate(self, prompt: str) -> str:
        try:
            content = self.client.invoke(prompt).content
        except (httpx.HTTPError, ResponseError, OSError) as exc:
            raise OllamaUnavailableError(
                "Ollama is unavailable. Start it with `ollama serve` and pull the configured model."
            ) from exc

        if isinstance(content, str):
            answer = content.strip()
        elif isinstance(content, list):
            answer = "".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        else:
            answer = ""
        if not answer:
            raise OllamaUnavailableError("Ollama returned an empty or unexpected response.")
        return answer


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

    def generate_with_mode(self, prompt: str) -> tuple[str, str]:
        try:
            return self.primary.generate(prompt), "local model"
        except OllamaUnavailableError:
            return self.fallback.generate(prompt), "offline excerpt"

    def generate(self, prompt: str) -> str:
        return self.generate_with_mode(prompt)[0]
