"""Lightweight query tokenization shared by retrieval and guided RAG."""

import re


def preprocess(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()
