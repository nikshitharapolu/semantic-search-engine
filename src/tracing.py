"""Small JSON views for opt-in LangSmith traces (never serialize engine objects)."""
from langsmith import traceable


def inputs_without_self(inputs):
    return {key: value for key, value in inputs.items() if key != "self"}


def response_view(response):
    if response is None:
        return {}
    result = {"answer": response.answer, "sources": response.sources.to_dict(orient="records")}
    for key in ("outcome", "trace"):
        if hasattr(response, key):
            result[key] = getattr(response, key)
    return result


def retrieval_view(frame):
    return {"documents": [
        {"page_content": str(row.get("text", "")),
         "metadata": {key: str(row.get(key, "")) for key in ("doc_id", "title", "source")}}
        for row in frame.to_dict(orient="records")
    ]}
