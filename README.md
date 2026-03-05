# Australian Defence Contracts — Natural Language Search

Search 216 Australian defence procurement contracts using natural language questions. Built with semantic search (FAISS + sentence-transformers) and optional AI-powered answers (Claude).

## Live Demo

[**Launch the app on Streamlit Cloud**](https://share.streamlit.io/)

## Features

- **Semantic search**: Ask questions in plain English — the system finds relevant contracts by meaning, not just keywords
- **1,038 searchable passages**: Contract metadata + tender documents from AusTender, indexed with `all-MiniLM-L6-v2` embeddings
- **216 contracts** from BIP Intelligence, covering construction, defence equipment, IT, services, and more
- **Optional AI answers**: Add your Anthropic API key to get Claude-generated summaries with contract citations
- **Colour-coded results** by buyer department (DSRG, CASG, DHA, ISG)

## Data Source

All contract data scraped from [BIP Intelligence](https://app.bipintelligence.com) (Australian defence procurement notices). Tender documents sourced from AusTender.

## Architecture

Adapted from [pitt-llm-workbench](https://github.com/Polly25-creator/pitt-llm-workbench):

1. **Chunk**: Each contract split into ~1000-char passages with metadata
2. **Embed**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
3. **Index**: FAISS `IndexFlatIP` for cosine similarity search
4. **Search**: Query → embed → FAISS top-k → display results
5. **RAG** (optional): Top results + question → Claude → natural language answer

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Example Questions

- What construction projects are happening in Queensland?
- Which contracts involve cybersecurity or IT services?
- Are there any contracts for naval vessel maintenance?
- What contracts does Defence Housing Australia have open?
- What opportunities exist for communications equipment?
