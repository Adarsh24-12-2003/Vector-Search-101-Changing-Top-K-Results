# Mini Vector Search

A small semantic-search project based on the Vector Search 101 template. It
embeds support articles, searches them with FAISS, and lets you choose how many
matches to display with `--top-k`.

## Setup

In PowerShell, from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in `.env`. The optional
`OPENAI_EMBEDDING_MODEL` defaults to `text-embedding-3-small`. Do not share or
commit `.env`.

## Run

```powershell
python test_code.py
python vector_search_demo.py
python vector_search_demo.py --query "I was charged after cancelling" --top-k 5
```

`--top-k` defaults to 5. If it is larger than the corpus, all available
articles are shown. A demo run sends the corpus and query to the configured
OpenAI embeddings model.

## Project files

- `vector_search_demo.py` - command-line app (`--query` and `--top-k`).
- `utils.py` - CSV loading, embedding, FAISS, and result formatting helpers.
- `test_code.py` - offline checks; it makes no API calls.
- `data/support_articles.csv` - search corpus.
- `.env.sample` - safe environment-variable template.
