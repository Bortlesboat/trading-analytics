"""
QUERY.PY - Search Your Documents
=================================
This script searches your indexed documents and (optionally) uses Claude to answer.

HOW IT WORKS:
1. Take your question
2. Convert it to embeddings (same as we did for documents)
3. Find the most similar chunks in the database
4. (Optional) Send chunks + question to Claude for a synthesized answer

You can run this WITHOUT an API key to just see the search results.
Add your Anthropic API key to get AI-generated answers.
"""

import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import paths from config.py (local) or use defaults
try:
    from config import DB_FOLDER
except ImportError:
    DB_FOLDER = "./faiss_db"

# How many chunks to retrieve (more = more context, but also more noise)
TOP_K = 5

# Set your Anthropic API key here OR as environment variable ANTHROPIC_API_KEY
# Get one at: https://console.anthropic.com/
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# =============================================================================
# STEP 1: Load the same embedding model used for indexing
# =============================================================================
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


# =============================================================================
# STEP 2: Load the FAISS index and metadata
# =============================================================================
print("Loading database...")

try:
    # Load FAISS index
    index = faiss.read_index(os.path.join(DB_FOLDER, "index.faiss"))

    # Load chunks and metadata
    with open(os.path.join(DB_FOLDER, "chunks.pkl"), 'rb') as f:
        chunks = pickle.load(f)

    with open(os.path.join(DB_FOLDER, "metadatas.pkl"), 'rb') as f:
        metadatas = pickle.load(f)

    print(f"Loaded index with {index.ntotal} vectors\n")

except FileNotFoundError:
    print("ERROR: No database found. Run ingest.py first!")
    exit(1)


# =============================================================================
# STEP 3: Search function
# =============================================================================
def search(query: str, n_results: int = TOP_K) -> dict:
    """
    Search the database for chunks similar to the query.

    Returns dict with:
    - documents: the actual text chunks
    - metadatas: info about where each chunk came from
    - distances: how similar (lower = more similar for L2 distance)
    """
    # Convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Search FAISS - returns distances and indices
    distances, indices = index.search(query_embedding, n_results)

    # Gather results
    results = {
        'documents': [chunks[i] for i in indices[0]],
        'metadatas': [metadatas[i] for i in indices[0]],
        'distances': distances[0].tolist()
    }

    return results


# =============================================================================
# STEP 4: Display results
# =============================================================================
def display_results(results: dict, query: str):
    """Pretty print the search results."""
    print("=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    documents = results['documents']
    metas = results['metadatas']
    distances = results['distances']

    for i, (doc, meta, dist) in enumerate(zip(documents, metas, distances)):
        # Convert L2 distance to similarity score (approximate)
        # Lower distance = more similar, so we invert it
        similarity = 1 / (1 + dist)
        print(f"\n--- Result {i+1} (similarity: {similarity:.2%}) ---")
        print(f"Source: {meta['source']}")
        print(f"Content: {doc[:300]}...")  # First 300 chars

    print("\n" + "=" * 60)


# =============================================================================
# STEP 5: (Optional) Get AI-generated answer
# =============================================================================
def get_ai_answer(query: str, results: dict) -> str:
    """Use Claude to synthesize an answer from the retrieved chunks."""

    if not ANTHROPIC_API_KEY:
        return None

    try:
        import anthropic

        # Combine retrieved chunks into context
        documents = results['documents']
        metas = results['metadatas']

        context_parts = []
        for doc, meta in zip(documents, metas):
            context_parts.append(f"[From: {meta['source']}]\n{doc}")

        context = "\n\n---\n\n".join(context_parts)

        # Create the prompt
        prompt = f"""Based on the following excerpts from my notes, answer this question: {query}

CONTEXT FROM MY NOTES:
{context}

INSTRUCTIONS:
- Only use information from the provided context
- If the context doesn't contain enough info, say so
- Be concise and direct
- Cite which source(s) you're drawing from

ANSWER:"""

        # Call Claude
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    except ImportError:
        return "Install anthropic package: pip install anthropic"
    except Exception as e:
        return f"Error calling Claude: {e}"


# =============================================================================
# MAIN: Interactive query loop
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG QUERY SYSTEM")
    print("=" * 60)
    print("Type a question to search your notes.")
    print("Type 'quit' to exit.\n")

    if ANTHROPIC_API_KEY:
        print("API key found - will generate AI answers")
    else:
        print("No API key - showing raw search results only")
        print("(Set ANTHROPIC_API_KEY env var for AI answers)\n")

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        # Search
        results = search(query)

        # Display raw results
        display_results(results, query)

        # Get AI answer if API key is set
        if ANTHROPIC_API_KEY:
            print("\nAI ANSWER:")
            print("-" * 40)
            answer = get_ai_answer(query, results)
            print(answer)
