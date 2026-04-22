import os
import sys
import faiss
import pickle
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG — edit as needed
# =========================

INDEX_FILE = "faiss_index.bin"
ID_MAP_FILE = "id_map.pkl"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
TOP_K = 8
MAX_CONTEXT_CHARS = 9000

# =========================
# STARTUP CHECKS
# =========================

if not os.path.exists(INDEX_FILE):
    print(f"ERROR: '{INDEX_FILE}' not found. Run build_index.py first.")
    sys.exit(1)

if not os.path.exists(ID_MAP_FILE):
    print(f"ERROR: '{ID_MAP_FILE}' not found. Run build_index.py first.")
    sys.exit(1)

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

# =========================
# LOAD FAISS + METADATA
# =========================

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

with open(ID_MAP_FILE, "rb") as f:
    id_map = pickle.load(f)

print(f"Index loaded: {index.ntotal} vectors")

# =========================
# RETRIEVAL FUNCTION
# =========================

def retrieve(query):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    distances, indices = index.search(query_embedding, TOP_K)

    results = []
    sources = []
    seen_urls = set()

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        if idx >= len(id_map):
            continue

        entry = id_map[idx]

        # Use the stored chunk text directly — no DB re-query needed
        chunk_text = entry["text"]
        url = entry["url"]

        results.append(chunk_text)

        if url not in seen_urls:
            sources.append(url)
            seen_urls.add(url)

    return results, sources

# =========================
# MAIN LOOP
# =========================

print("\nGNDEC RAG ready. Type your question (or 'exit' to quit).\n")

while True:
    try:
        query = input("Ask: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not query:
        continue

    if query.lower() in ["exit", "quit"]:
        break

    print("Retrieving context...")

    retrieved_chunks, sources = retrieve(query)

    if not retrieved_chunks:
        print("\nAnswer: No relevant information found in GNDEC data.\n")
        continue

    context = "\n\n".join(retrieved_chunks)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""You are a helpful assistant answering questions about GNDEC (Guru Nanak Dev Engineering College).
Use the provided context to answer the question. If the context contains relevant information, use it.
If the context does not contain enough information, answer based on your general knowledge about GNDEC.
Keep your answer clear and concise.

Context:
{context}

Question: {query}

Answer:"""

    print("Sending to LLM...")

    try:
        response = requests.post(
            LLAMA_URL,
            json={
                "model": "Llama-3.2-3B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        if "choices" not in data:
            print("LLM Error:", data)
            continue

        answer = data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to LLM server at", LLAMA_URL)
        print("Make sure your llama-server is running.")
        continue
    except requests.exceptions.Timeout:
        print("ERROR: LLM request timed out.")
        continue
    except Exception as e:
        print("ERROR:", e)
        continue

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for src in sources:
        print(" -", src)

    print()
