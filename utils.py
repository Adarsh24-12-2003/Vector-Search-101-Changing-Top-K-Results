"""Small helpers for loading support articles and searching them with FAISS."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv


def load_articles(csv_path: Path) -> list[dict[str, str]]:
    """Load support articles and create one searchable string per article.

    Args:
        csv_path: CSV file containing doc_id, title, category, text, and keywords.

    Returns:
        Article dictionaries with an extra ``searchable_text`` field.
    """
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        articles = list(csv.DictReader(file))

    required = {"doc_id", "title", "category", "text", "keywords"}
    if not articles or not required.issubset(articles[0]):
        raise ValueError(f"{csv_path} must contain these columns: {sorted(required)}")

    for article in articles:
        article["searchable_text"] = " ".join(
            [article["title"], article["category"], article["text"], article["keywords"]]
        )
    return articles


def create_embeddings(texts: Iterable[str]) -> np.ndarray:
    """Create normalized OpenAI embeddings for documents or queries.

    The function loads ``OPENAI_API_KEY`` from this folder's ``.env`` file.
    ``OPENAI_EMBEDDING_MODEL`` is optional and defaults to
    ``text-embedding-3-small``.

    Args:
        texts: Strings to embed in one batched API request.

    Returns:
        A two-dimensional float32 NumPy array with unit-length rows.
    """
    from openai import OpenAI

    text_list = [str(text) for text in texts]
    if not text_list:
        raise ValueError("At least one text is required.")

    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Add OPENAI_API_KEY to the .env file before running the demo.")

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    response = OpenAI(api_key=api_key).embeddings.create(model=model, input=text_list)
    vectors = np.asarray([item.embedding for item in response.data], dtype="float32")
    return normalize_vectors(vectors)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return a float32 copy whose rows have length one.

    Args:
        vectors: One vector or a matrix of vectors.

    Returns:
        A two-dimensional, L2-normalized NumPy array.
    """
    matrix = np.asarray(vectors, dtype="float32").copy()
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("Zero-length vectors cannot be normalized.")
    return matrix / lengths


def build_faiss_index(document_vectors: np.ndarray):
    """Build an exact FAISS inner-product index from normalized vectors.

    Inner product on unit-length vectors is cosine similarity. ``IndexFlatIP``
    is exact and easy to explain, which makes it suitable for this small lab.

    Args:
        document_vectors: Normalized document embedding matrix.

    Returns:
        A populated ``faiss.IndexFlatIP`` instance.
    """
    import faiss

    vectors = normalize_vectors(document_vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def dense_search(
    query_vector: np.ndarray,
    articles: list[dict[str, str]],
    index: Any,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search FAISS and map its row numbers back to readable articles.

    Args:
        query_vector: Embedding for one query.
        articles: Article records in the same order as the FAISS vectors.
        index: Populated FAISS index.
        top_k: Maximum number of results to return.

    Returns:
        Ranked result dictionaries with cosine-similarity scores.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    result_count = min(top_k, len(articles), index.ntotal)
    if result_count == 0:
        raise ValueError("Search requires at least one indexed article.")
    scores, indices = index.search(normalize_vectors(query_vector), result_count)
    return _format_results(articles, indices[0], scores[0])


def print_results(title: str, results: list[dict[str, Any]]) -> None:
    """Print compact ranked results for a live class demo.

    Args:
        title: Heading printed above the results.
        results: Rows returned by :func:`dense_search`.
    """
    print(f"\n{title}")
    print("-" * len(title))
    for result in results:
        print(
            f"{result['rank']}. {result['doc_id']} | "
            f"score={result['score']:.3f} | {result['title']}"
        )


def _format_results(
    articles: list[dict[str, str]], indices: Iterable[int], scores: Iterable[float]
) -> list[dict[str, Any]]:
    """Convert numeric search output into student-readable dictionaries."""
    results = []
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        article = articles[int(index)]
        results.append(
            {
                "rank": rank,
                "doc_id": article["doc_id"],
                "title": article["title"],
                "category": article["category"],
                "score": float(score),
            }
        )
    return results
