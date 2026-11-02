# Guided RAG evidence checks, version 2

Verified October 31, 2026, using the existing 18-question set unchanged.
The first 18 questions were used to diagnose and tune the implementation. They
are now development data, not an independent held-out benchmark.

## Changes

Topical similarity does not establish that a document contains a requested fact.
The guided evidence gate now checks for dates, quantities and attribution text
when the question requests those types of facts. These deterministic checks are
heuristics, not entailment proofs or universal factuality validation.

Evidence selection promotes a candidate only when both original-question term
overlap and semantic alignment improve over the first result. A retry cannot
replace original results with results that have worse original-question term
coverage. Relevance thresholds were adjusted to avoid rejecting paraphrases.
No question IDs, expected document IDs or evaluation labels enter the pipeline.

## Reproduced outcomes

On the unchanged development set, original and both guided versions retrieved
and cited the expected document on all 12 answerable cases. The earlier guided
version abstained on three of six unanswerable cases. Version 2 abstains on all
six, without rejecting any of the 12 answerable cases. Original abstains on none.

The 12 additional diagnostic questions exposed false abstentions during
iteration and therefore also informed development. They are NOT held-out data.
After the fixes, guided cites the expected document on all six positive cases,
versus five for original, and abstains on all six negative cases, versus none for
original. A new independently authored evaluation set is needed to measure
whether these changes generalize beyond both development sets.

## Scoring correction

Previously `valid_citations` assigned zero to a correctly abstained answer,
which makes its aggregate an unsuitable measure of citation quality. Fresh
experiments apply the SAME revised evaluators to both workflows:
`answered_valid_citations` is scored only when an answer is returned; abstention
is scored separately, while source-hit and expected-source-citation metrics on
answerable cases still penalize false abstentions. Old scores are retained as
history and should not be compared directly with the renamed citation metric.

All evaluation responses used extractive evidence, not LLM generation. The guided
path remains slower because it does additional checks and sometimes a second
retrieval; no latency improvement is claimed. Absolute latency on this tiny
cached corpus is not a production performance benchmark. All 36 unit/API tests
passed. Per-question before/after results are included alongside this document.

LangSmith experiments whose names include `evidence-checks-v2` contain the new
runs. The original 18 labels and corpus were not changed. Initial and prior
experiments remain available for audit. AWS compute was not restarted.
