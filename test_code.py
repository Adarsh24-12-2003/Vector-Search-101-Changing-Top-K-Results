"""Run fast offline checks for the mini vector-search app.

This test does not call OpenAI. It validates data loading and Top-K FAISS
search using tiny fixed vectors.
"""

from pathlib import Path

import numpy as np

from utils import (
    build_faiss_index,
    dense_search,
    load_articles,
)


def main() -> None:
    """Execute the offline checks and raise an error if any check fails."""
    data_path = Path(__file__).parent / "data" / "support_articles.csv"
    articles = load_articles(data_path)

    sample_articles = articles[:6]
    document_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    index = build_faiss_index(document_vectors)
    dense_results = dense_search(np.array([1.0, 0.0, 0.0]), sample_articles, index)
    assert dense_results[0]["doc_id"] == sample_articles[0]["doc_id"]
    assert len(dense_results) == 5
    assert len(
        dense_search(np.array([1.0, 0.0, 0.0]), sample_articles, index, top_k=10)
    ) == len(sample_articles)
    try:
        dense_search(np.array([1.0, 0.0, 0.0]), sample_articles, index, top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("top_k=0 should be rejected")

    print("All mini vector-search offline checks passed.")


if __name__ == "__main__":
    main()
