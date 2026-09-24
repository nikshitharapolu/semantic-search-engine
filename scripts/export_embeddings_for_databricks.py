from __future__ import annotations
import pandas as pd
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from service import get_engine
from utils import MODEL_NAME


def main() -> None:
    engine = get_engine()
    output_path = Path("/app/data/databricks_embeddings.csv")

    export_df = engine.df.copy()

    export_df["embedding"] = [
        json.dumps(vector.astype(float).tolist())
        for vector in engine.doc_embeddings
    ]
    export_df["embedding_model"] = MODEL_NAME
    export_df["embedding_dimensions"] = engine.doc_embeddings.shape[1]

    columns = [
        "doc_id",
        "title",
        "category",
        "text",
        "embedding",
        "embedding_model",
        "embedding_dimensions",
    ]

    export_df[columns].to_csv(output_path, index=False)

    print(f"Exported rows: {len(export_df)}")
    print(f"Embedding dimensions: {engine.doc_embeddings.shape[1]}")
    print(f"Output: {output_path}")
    queries_path = Path("/app/data/queries.csv")
    query_output_path = Path(
        "/app/data/databricks_query_embeddings.csv"
    )

    queries_df = pd.read_csv(queries_path)

    query_vectors = engine.model.encode(
        queries_df["query"].astype(str).tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    queries_df["query_embedding"] = [
        json.dumps(vector.astype(float).tolist())
        for vector in query_vectors
    ]
    queries_df["embedding_model"] = MODEL_NAME
    queries_df["embedding_dimensions"] = query_vectors.shape[1]

    queries_df.to_csv(query_output_path, index=False)

    print(f"Exported queries: {len(queries_df)}")
    print(f"Query output: {query_output_path}")


if __name__ == "__main__":
    main()