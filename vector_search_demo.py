"""Run a compact semantic-search demo with a configurable result count."""

import argparse
from pathlib import Path

from utils import (
    build_faiss_index,
    create_embeddings,
    dense_search,
    load_articles,
    print_results,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_QUERY = "How can I get my money back after cancelling?"
DEFAULT_TOP_K = 5


def main() -> None:
    """Embed the corpus and query, then print the nearest matching articles."""
    parser = argparse.ArgumentParser(
        description="Search support articles and choose how many results to show."
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"Search text (default: {DEFAULT_QUERY!r})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help=f"Maximum results to show (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    query = args.query.strip()
    if not query:
        parser.error("--query cannot be empty")

    articles = load_articles(DATA_DIR / "support_articles.csv")
    print(f"Loaded {len(articles)} support articles.")

    texts = [article["searchable_text"] for article in articles]
    vectors = create_embeddings([*texts, query])
    document_vectors = vectors[:-1]
    query_vector = vectors[-1]
    faiss_index = build_faiss_index(document_vectors)
    results = dense_search(query_vector, articles, faiss_index, top_k=args.top_k)
    print_results(f"Top {len(results)} matches for: {query}", results)


if __name__ == "__main__":
    main()
