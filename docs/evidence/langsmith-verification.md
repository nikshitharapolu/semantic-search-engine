# LangSmith verification — October 31, 2026

Agent: `hybrid-search-rag`. Interactive test tracing used the Local environment;
evaluation runs were stored under their experiment destinations. Retrieved text,
answers, nested graph nodes and retrieval spans were confirmed in LangSmith.

The fixed dataset contains 18 labeled questions. Two completed experiments each
have 18 persisted runs, zero application errors, and evaluator feedback on every
run. No paid model API or LLM judge was called. The original path produced 18
extractive excerpts; the guided path produced 15 excerpts and three explicit
abstentions. This demonstrates tracing/evaluation wiring on the demo corpus;
it does not establish generated-answer quality or production reliability.

- [Original experiment](https://smith.langchain.com/o/ff36c5fa-462c-4401-896a-e422578fe702/datasets/3c4fce0a-8b38-4be6-88a2-d45f9caf7448/compare?selectedSessions=e49c07d1-4be9-400d-b5de-02f807fda428)
- [Guided experiment](https://smith.langchain.com/o/ff36c5fa-462c-4401-896a-e422578fe702/datasets/3c4fce0a-8b38-4be6-88a2-d45f9caf7448/compare?selectedSessions=a65d96ad-1336-4888-b246-6aa023d18ad0)

Code-based checks cover citation validity, expected-source citation, source hit,
abstention matching labels and offline-excerpt mode. Latency is recorded by the
traces. All 30 local tests passed. Earlier experiments with SDK setup errors
remain as debugging history and should not be used as completed results.

The private `.env` is ignored by Git, and tracing remains disabled there by
default. Live runs explicitly enable tracing for their process only. No AWS
compute was started for this verification. Rerun with Ollama for model-generated
answers and review answer correctness separately before making quality claims.
