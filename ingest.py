"""
INGEST.PY - Document Indexing Script
=====================================
This script reads your markdown files and stores them in a searchable database.

HOW IT WORKS:
1. Find all .md files in a folder
2. Read each file's content
3. Split into smaller chunks (LLMs work better with small pieces)
4. Convert each chunk into "embeddings" (numbers that represent meaning)
5. Store in FAISS (a vector database from Meta/Facebook)

WHY EMBEDDINGS?
- Computers can't understand text directly
- Embeddings convert "I love dogs" into something like [0.2, -0.5, 0.8, ...]
- Similar meanings = similar numbers
- "I love dogs" and "I adore puppies" would have very similar embeddings
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import paths from config.py (local) or use defaults
try:
    from config import DOCS_FOLDER, DB_FOLDER
except ImportError:
    DOCS_FOLDER = "./docs"
    DB_FOLDER = "./faiss_db"

# How many characters per chunk (smaller = more precise, larger = more context)
CHUNK_SIZE = 500

# How many characters to overlap between chunks (prevents cutting mid-thought)
CHUNK_OVERLAP = 50


# =============================================================================
# STEP 1: Load the embedding model
# =============================================================================
print("Loading embedding model (first time will download ~90MB)...")

# This model is small, fast, and good for general text
# It runs entirely on your computer - no API calls
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Model loaded!")


# =============================================================================
# STEP 2: Create output folder
# =============================================================================
os.makedirs(DB_FOLDER, exist_ok=True)
print(f"\nDatabase will be saved to: {DB_FOLDER}")


# =============================================================================
# STEP 3: Helper function to chunk text
# =============================================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.

    Why overlap? If a sentence is "The quick brown fox jumps over the lazy dog"
    and we cut at "fox", we lose the connection between fox and jumps.
    Overlap ensures we capture context across chunk boundaries.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # Move forward, but overlap with previous chunk
        start = end - overlap

    return chunks


# =============================================================================
# STEP 4: Find and process all markdown files
# =============================================================================
print(f"\nScanning for .md files in: {DOCS_FOLDER}")

docs_path = Path(DOCS_FOLDER)
md_files = list(docs_path.rglob("*.md"))  # rglob = recursive search

print(f"Found {len(md_files)} markdown files")

# Limit to first 20 files for initial testing (faster)
MAX_FILES = 20
files_to_process = md_files[:MAX_FILES]

print(f"Processing first {len(files_to_process)} files for testing...\n")


# =============================================================================
# STEP 5: Process each file
# =============================================================================
all_chunks = []      # The actual text chunks
all_metadatas = []   # Info about where each chunk came from

for i, filepath in enumerate(files_to_process):
    try:
        # Read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip very short files
        if len(content) < 50:
            continue

        # Get relative path for cleaner display
        relative_path = filepath.relative_to(docs_path)

        # Chunk the content
        chunks = chunk_text(content)

        # Add each chunk with metadata
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": str(relative_path),
                "chunk_index": j
            })

        print(f"  [{i+1}/{len(files_to_process)}] {relative_path}: {len(chunks)} chunks")

    except Exception as e:
        print(f"  Error reading {filepath}: {e}")


# =============================================================================
# STEP 6: Generate embeddings
# =============================================================================
print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")

# Convert all chunks to embeddings in one batch (faster than one at a time)
embeddings = model.encode(all_chunks, show_progress_bar=True)

# Convert to numpy array and ensure float32 (FAISS requirement)
embeddings_array = np.array(embeddings).astype('float32')

print(f"Embeddings shape: {embeddings_array.shape}")
# Shape is (num_chunks, 384) - 384 is the embedding dimension for this model


# =============================================================================
# STEP 7: Create FAISS index and add embeddings
# =============================================================================
print("\nCreating FAISS index...")

# Get the dimension of our embeddings
dimension = embeddings_array.shape[1]

# Create a simple flat index (exact search, good for small datasets)
# For larger datasets, you'd use IndexIVFFlat or IndexHNSW for faster search
index = faiss.IndexFlatL2(dimension)

# Add our embeddings to the index
index.add(embeddings_array)

print(f"Index created with {index.ntotal} vectors")


# =============================================================================
# STEP 8: Save everything to disk
# =============================================================================
print("\nSaving to disk...")

# Save the FAISS index
faiss.write_index(index, os.path.join(DB_FOLDER, "index.faiss"))

# Save the chunks and metadata (FAISS only stores vectors, not the text)
with open(os.path.join(DB_FOLDER, "chunks.pkl"), 'wb') as f:
    pickle.dump(all_chunks, f)

with open(os.path.join(DB_FOLDER, "metadatas.pkl"), 'wb') as f:
    pickle.dump(all_metadatas, f)

print(f"\nDone! Indexed {len(all_chunks)} chunks from {len(files_to_process)} files.")
print(f"\nFiles saved:")
print(f"  - {DB_FOLDER}/index.faiss (the vector index)")
print(f"  - {DB_FOLDER}/chunks.pkl (the text content)")
print(f"  - {DB_FOLDER}/metadatas.pkl (source file info)")
print("\nNext step: Run query.py to search your notes!")
