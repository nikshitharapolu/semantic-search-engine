from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generation import OllamaGenerator, OllamaUnavailableError, ResilientLocalGenerator


class LangChainOllamaTests(unittest.TestCase):
    @patch("generation.ChatOllama")
    def test_model_uses_langchain_with_local_configuration(self, chat_model):
        chat_model.return_value.invoke.return_value = SimpleNamespace(
            content="Supported answer [1]."
        )
        generator = OllamaGenerator(model="llama3.2:3b", base_url="http://localhost:11434")

        self.assertEqual(generator.generate("Question and context"), "Supported answer [1].")
        chat_model.assert_called_once_with(
            model="llama3.2:3b", base_url="http://localhost:11434", temperature=0.1,
            client_kwargs={"timeout": 120, "trust_env": False},
        )
        chat_model.return_value.invoke.assert_called_once_with("Question and context")

    @patch("generation.ChatOllama")
    def test_unavailable_local_server_uses_the_existing_offline_fallback(self, chat_model):
        chat_model.return_value.invoke.side_effect = httpx.ConnectError("offline")

        class Fallback:
            def generate(self, prompt: str) -> str:
                return "Local cited excerpt [1]"

        generator = ResilientLocalGenerator(OllamaGenerator(), Fallback())

        self.assertEqual(
            generator.generate_with_mode("Context"),
            ("Local cited excerpt [1]", "offline excerpt"),
        )

    @patch("generation.ChatOllama")
    def test_empty_model_response_is_treated_as_unavailable(self, chat_model):
        chat_model.return_value.invoke.return_value = SimpleNamespace(content="")

        with self.assertRaises(OllamaUnavailableError):
            OllamaGenerator().generate("Context")
