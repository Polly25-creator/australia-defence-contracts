#!/usr/bin/env python3
"""
Australian Defence Contracts — Natural Language Search & Q&A

Streamlit web app adapted from pitt-llm-workbench:
  - FAISS vector search for semantic similarity
  - sentence-transformers embeddings (all-MiniLM-L6-v2)
  - Optional LLM integration (Anthropic Claude) for RAG-powered answers
  - Falls back to search-only mode if no API key provided

Usage:
    streamlit run defence_search_app.py
"""

import json
import os
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "search_data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "faiss.index"
METADATA_JSON = BASE_DIR / "contracts_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load sentence-transformer model"""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_index():
    """Load FAISS index and chunks"""
    assert INDEX_FILE.exists(), f"Index not found at {INDEX_FILE}. Run build_index.py first."
    index = faiss.read_index(str(INDEX_FILE))
    chunks = []
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return index, chunks


@st.cache_data
def load_contracts():
    """Load full contract metadata"""
    with open(METADATA_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Search (following pitt-llm-workbench search_index pattern)
# ---------------------------------------------------------------------------
def search_index(query, top_k=10):
    """Semantic search using FAISS — same pattern as pitt-llm-workbench"""
    model = load_model()
    index, chunks = load_index()

    # Embed query
    q_emb = model.encode([query])[0].astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q_emb)

    # Search
    D, I = index.search(q_emb, min(top_k, len(chunks)))

    results = []
    seen_contracts = set()
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx].copy()
        chunk['score'] = float(score)
        results.append(chunk)
        seen_contracts.add(chunk.get('contract_id', ''))

    return results, len(seen_contracts)


def format_context(results, max_chars=6000):
    """Format search results into context for LLM (RAG pattern)"""
    context_parts = []
    chars = 0
    for r in results:
        block = f"[Contract: {r['title']} | Ref: {r['reference']} | Buyer: {r['buyer']} | Sector: {r['sector']}]\n{r['text']}"
        if chars + len(block) > max_chars:
            break
        context_parts.append(block)
        chars += len(block)
    return "\n\n---\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# LLM Answer Generation (RAG — same pattern as pitt-llm-workbench)
# ---------------------------------------------------------------------------
def generate_answer(query, results, api_key):
    """Use Claude to generate a natural language answer from search results"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        context = format_context(results)

        system_prompt = """You are an expert analyst of Australian defence procurement contracts.
You answer questions using ONLY the contract data provided in the context below.
Always cite specific contracts by their title and reference number.
If the context doesn't contain enough information to answer, say so.
Be direct and factual. Use plain English."""

        user_prompt = f"""Based on the following Australian defence contract data, answer this question:

QUESTION: {query}

CONTRACT DATA:
{context}

Provide a clear, well-structured answer citing specific contracts where relevant."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def render_result_card(result, rank):
    """Render a single search result as a card"""
    score = result.get('score', 0)
    title = result.get('title', 'Untitled')
    ref = result.get('reference', 'N/A')
    buyer = result.get('buyer', '')
    sector = result.get('sector', '')
    published = result.get('published', '')
    deadline = result.get('deadline', '')
    chunk_type = result.get('chunk_type', '')
    text = result.get('text', '')

    # Color by buyer
    buyer_colors = {
        'DSRG': '#1a5276',
        'CASG': '#1e8449',
        'Defence Housing Australia': '#b7950b',
        'ISG': '#6c3483'
    }
    buyer_short = buyer.split(' - ')[-1].strip() if ' - ' in buyer else buyer
    color = buyer_colors.get(buyer_short, '#2c3e50')

    # Score badge
    score_pct = min(int(score * 100), 100)

    with st.container():
        st.markdown(f"""
<div style="border-left: 4px solid {color}; padding: 12px 16px; margin: 8px 0;
            background: #f8f9fa; border-radius: 0 8px 8px 0;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="font-size: 1.05em; color: {color};">
            {rank}. {title}
        </strong>
        <span style="background: {'#27ae60' if score > 0.5 else '#f39c12' if score > 0.3 else '#95a5a6'};
              color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8em;">
            {score_pct}% match
        </span>
    </div>
    <div style="color: #666; font-size: 0.85em; margin: 4px 0;">
        <strong>Ref:</strong> {ref} &nbsp;|&nbsp;
        <strong>Buyer:</strong> {buyer_short} &nbsp;|&nbsp;
        <strong>Sector:</strong> {sector} &nbsp;|&nbsp;
        <em>{'📄 Contract Info' if chunk_type == 'contract_info' else '📋 Tender Document'}</em>
    </div>
    {"<div style='color: #666; font-size: 0.8em;'><strong>Published:</strong> " + published + " &nbsp;|&nbsp; <strong>Deadline:</strong> " + deadline + "</div>" if published else ""}
</div>""", unsafe_allow_html=True)

        with st.expander("View matched text"):
            st.text(text[:800] + ("..." if len(text) > 800 else ""))


def main():
    st.set_page_config(
        page_title="Defence Contracts Search",
        page_icon="🔍",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .stApp {max-width: 1200px; margin: 0 auto;}
    .main-header {text-align: center; padding: 20px 0;}
    .stat-box {
        background: linear-gradient(135deg, #1a5276, #2980b9);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; margin: 5px;
    }
    .stat-number {font-size: 2em; font-weight: bold;}
    .stat-label {font-size: 0.9em; opacity: 0.9;}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇦🇺 Australian Defence Contracts</h1>
        <p style="color: #666; font-size: 1.1em;">
            Natural language search across 216 contracts from BIP Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")

        # API key for LLM-powered answers
        api_key = st.text_input(
            "Anthropic API Key (optional)",
            type="password",
            help="Add your Claude API key to enable AI-generated answers. "
                 "Without it, you'll get search results only."
        )

        use_llm = bool(api_key)

        st.markdown("---")

        top_k = st.slider("Results to show", 5, 30, 10)

        st.markdown("---")

        # Stats
        contracts = load_contracts()
        _, chunks = load_index()

        st.markdown("### Database Stats")
        st.metric("Contracts", len(contracts))
        st.metric("Searchable Chunks", len(chunks))

        # Buyer breakdown
        buyers = {}
        for c in contracts:
            b = (c.get('awardingAuthority', '') or '').split(' - ')[-1].strip()
            buyers[b] = buyers.get(b, 0) + 1

        st.markdown("### By Buyer")
        for b, count in sorted(buyers.items(), key=lambda x: -x[1]):
            st.markdown(f"- **{b}**: {count}")

        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. Your question is converted to a vector embedding
        2. FAISS finds the most similar contract passages
        3. If an API key is provided, Claude synthesizes an answer

        *Based on [pitt-llm-workbench](https://github.com/Polly25-creator/pitt-llm-workbench) RAG architecture*
        """)

    # Example queries
    st.markdown("**Try asking:**")
    examples = [
        "What construction projects are happening in Queensland?",
        "Which contracts involve cybersecurity or IT services?",
        "What are the largest building refurbishment projects?",
        "Are there any contracts for naval vessel maintenance?",
        "What contracts does Defence Housing Australia have open?",
        "Which tenders have deadlines in April 2026?",
        "What opportunities exist for communications equipment?",
    ]

    cols = st.columns(3)
    for i, example in enumerate(examples[:6]):
        with cols[i % 3]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state['query'] = example

    # Search input
    query = st.text_input(
        "Ask a question about Australian defence contracts:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., What construction work is planned for RAAF bases?",
        key="search_input"
    )

    if query:
        with st.spinner("Searching contracts..."):
            results, n_contracts = search_index(query, top_k=top_k)

        if not results:
            st.warning("No results found. Try rephrasing your question.")
            return

        st.markdown(f"**Found {len(results)} matching passages across {n_contracts} contracts**")

        # LLM Answer (RAG)
        if use_llm:
            with st.spinner("Generating answer with Claude..."):
                answer = generate_answer(query, results, api_key)

            st.markdown("### AI Answer")
            st.markdown(f"""
            <div style="background: #eaf2f8; border: 1px solid #aed6f1; border-radius: 12px;
                        padding: 20px; margin: 16px 0;">
                {answer}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

        # Search Results
        st.markdown("### Search Results")

        # Tabs for different views
        tab1, tab2 = st.tabs(["All Results", "By Contract"])

        with tab1:
            for i, result in enumerate(results):
                render_result_card(result, i + 1)

        with tab2:
            # Group by contract
            by_contract = {}
            for r in results:
                cid = r.get('contract_id', '')
                if cid not in by_contract:
                    by_contract[cid] = {'title': r['title'], 'ref': r['reference'],
                                         'buyer': r['buyer'], 'sector': r['sector'],
                                         'chunks': [], 'best_score': 0}
                by_contract[cid]['chunks'].append(r)
                by_contract[cid]['best_score'] = max(by_contract[cid]['best_score'], r['score'])

            sorted_contracts = sorted(by_contract.values(), key=lambda x: -x['best_score'])

            for contract_group in sorted_contracts:
                score_pct = min(int(contract_group['best_score'] * 100), 100)
                with st.expander(
                    f"**{contract_group['title']}** — {contract_group['ref']} "
                    f"({score_pct}% match, {len(contract_group['chunks'])} passages)"
                ):
                    st.markdown(f"**Buyer:** {contract_group['buyer']} | **Sector:** {contract_group['sector']}")
                    for chunk in contract_group['chunks']:
                        st.markdown(f"*{'Contract Info' if chunk['chunk_type'] == 'contract_info' else 'Tender Document'}* "
                                    f"(score: {chunk['score']:.2f})")
                        st.text(chunk['text'][:600])
                        st.markdown("---")


if __name__ == "__main__":
    main()

