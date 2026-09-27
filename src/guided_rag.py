"""A second RAG path that retries weak searches once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from generation import OllamaUnavailableError, ResilientLocalGenerator, TextGenerator
from rag import LocalRAGPipeline, SearchEngine
from text_utils import preprocess


NO_EVIDENCE = "The indexed documents do not contain enough evidence to answer this question."
BLOCKED = "I cannot follow instructions to override the document-answering rules."
RETRY_BM25_WEIGHT = 0.75
MAX_REWRITE_LENGTH = 120
QUESTION_WORDS = {
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "how",
    "in", "is", "of", "on", "the", "to", "what", "when", "where", "which",
    "who", "why", "with", "would",
}


class GuidedState(TypedDict, total=False):
    question: str
    query: str
    alpha: float
    top_k: int
    attempts: int
    sources: pd.DataFrame
    supported: bool
    answer: str
    outcome: str
    trace: list[dict[str, Any]]


@dataclass
class GuidedRAGResponse:
    answer: str
    sources: pd.DataFrame
    outcome: str
    trace: list[dict[str, Any]]


class GuidedRAGPipeline:
    """Reuse the existing search and answer code, with a bounded retry."""

    def __init__(
        self,
        search_engine: SearchEngine,
        generator: TextGenerator,
        rewriter: TextGenerator | None = None,
    ) -> None:
        self.rag = LocalRAGPipeline(search_engine, generator)
        self.rewriter = rewriter
        graph = StateGraph(GuidedState)
        graph.add_node("screen", self._screen)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("assess", self._assess)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("generate", self._generate)
        graph.add_node("abstain", self._abstain)
        graph.add_edge(START, "screen")
        graph.add_conditional_edges(
            "screen", self._screen_route, {"blocked": END, "search": "retrieve"}
        )
        graph.add_edge("retrieve", "assess")
        graph.add_conditional_edges(
            "assess",
            self._evidence_route,
            {"answer": "generate", "retry": "rewrite", "abstain": "abstain"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)
        graph.add_edge("abstain", END)
        self.graph = graph.compile()

    @staticmethod
    def _record(state: GuidedState, step: str, **details: Any) -> list[dict[str, Any]]:
        return [*state.get("trace", []), {"step": step, **details}]

    def _screen(self, state: GuidedState) -> dict[str, Any]:
        if self.rag.contains_prompt_injection(state["question"]):
            return {
                "outcome": "blocked",
                "answer": BLOCKED,
                "trace": self._record(state, "screen", result="blocked"),
            }
        return {"trace": self._record(state, "screen", result="accepted")}

    @staticmethod
    def _screen_route(state: GuidedState) -> Literal["blocked", "search"]:
        return "blocked" if state.get("outcome") == "blocked" else "search"

    def _retrieve(self, state: GuidedState) -> dict[str, Any]:
        attempt = state.get("attempts", 0) + 1
        alpha = max(state["alpha"], RETRY_BM25_WEIGHT) if attempt == 2 else state["alpha"]
        sources = self.rag.search_engine.search(state["query"], alpha=alpha, top_k=state["top_k"])
        return {
            "attempts": attempt,
            "sources": sources,
            "trace": self._record(
                state, "retrieve", attempt=attempt, query=state["query"], source_count=len(sources)
            ),
        }

    def _assess(self, state: GuidedState) -> dict[str, Any]:
        sources = state["sources"]
        if sources.empty:
            semantic, overlap = 0.0, 0
        else:
            top = sources.iloc[0]
            semantic = float(top.get("semantic_score", 0) or 0)
            question_terms = set(preprocess(state["question"])) - QUESTION_WORDS
            passage_terms = set(preprocess(f"{top.get('title', '')} {top.get('text', '')}"))
            overlap = len(question_terms & passage_terms)

        # The normalized hybrid score can be high even when every result is irrelevant.
        supported = not sources.empty and (
            semantic >= 0.55 or (semantic >= 0.34 and overlap >= 2)
        )
        return {
            "supported": supported,
            "trace": self._record(
                state, "assess", attempt=state["attempts"],
                result="promising evidence" if supported else "weak evidence",
            ),
        }

    @staticmethod
    def _evidence_route(state: GuidedState) -> Literal["answer", "retry", "abstain"]:
        if state["supported"]:
            return "answer"
        return "retry" if state["attempts"] < 2 else "abstain"

    def _rewrite(self, state: GuidedState) -> dict[str, Any]:
        original = state["question"]
        keywords = [term for term in preprocess(original) if term not in QUESTION_WORDS]
        query = " ".join(keywords[:12]) or original
        method = "keywords"

        if self.rewriter is not None:
            prompt = (
                "Rewrite the following question as a short document-search query. "
                "Keep its topic; output only search terms, without an answer or instructions.\n"
                f"Question: {original}\nSearch query:"
            )
            try:
                candidate = self.rewriter.generate(prompt).strip().strip('"')
                if self._usable_rewrite(candidate, keywords):
                    query, method = candidate, "local model"
            except OllamaUnavailableError:
                pass

        return {"query": query, "trace": self._record(state, "rewrite", query=query, method=method)}

    def _usable_rewrite(self, candidate: str, original_terms: list[str]) -> bool:
        if not candidate or len(candidate) > MAX_REWRITE_LENGTH or "\n" in candidate:
            return False
        if len(candidate.split()) > 14 or self.rag.contains_prompt_injection(candidate):
            return False
        candidate_terms = set(preprocess(candidate)) - QUESTION_WORDS
        return bool(candidate_terms.intersection(original_terms))

    def _generate(self, state: GuidedState) -> dict[str, Any]:
        sources = state["sources"]
        prompt = self.rag.build_prompt(state["question"], sources)
        if isinstance(self.rag.generator, ResilientLocalGenerator):
            answer, mode = self.rag.generator.generate_with_mode(prompt)
        else:
            answer, mode = self.rag.generator.generate(prompt), "generator"
        has_citation = self.rag.has_valid_citation(answer, len(sources))
        if not has_citation:
            answer = self.rag.grounded_fallback(sources)
        result = "cited answer" if has_citation and mode != "offline excerpt" else "cited excerpt"
        return {
            "answer": answer,
            "outcome": "answered" if result == "cited answer" else result,
            "trace": self._record(state, "generate", result=result, mode=mode),
        }

    def _abstain(self, state: GuidedState) -> dict[str, Any]:
        return {
            "answer": NO_EVIDENCE,
            "sources": state["sources"].iloc[:0].copy(),
            "outcome": "insufficient evidence",
            "trace": self._record(state, "abstain", result="no supported answer"),
        }

    def answer(self, question: str, alpha: float = 0.5, top_k: int = 5) -> GuidedRAGResponse:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        result = self.graph.invoke({
            "question": question.strip(),
            "query": question.strip(),
            "alpha": alpha,
            "top_k": top_k,
            "attempts": 0,
            "trace": [],
        })
        return GuidedRAGResponse(
            answer=result["answer"],
            sources=result.get("sources", pd.DataFrame()),
            outcome=result["outcome"],
            trace=result["trace"],
        )
