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
TOP_K = 7
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
    query = query.lower()
    if "principle" in query:
        query = query.replace("principle", "principal")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    distances, indices = index.search(query_embedding, TOP_K)

    scored_entries = []

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        if idx >= len(id_map):
            continue

        entry = id_map[idx]
        url = entry.get("url", "")

        # HARD FILTER: skip PDFs completely
        if ".pdf" in url.lower():
            continue

        score = float(distances[0][i])
        
        priority = entry.get("priority", "medium")
        source_type = entry.get("source_type", "")

        # Priority weighting
        if priority == "high":
            score = score * 1.3
        elif priority == "low":
            score = score * 0.6
        else:
            score = score * 1.0

        if source_type == "pdf":
            score = score * 0.6
            
        if entry["url"] == "manual_entry":
            score = score * 2.0
            
        scored_entries.append({
            "score": score,
            "text": entry["text"],
            "url": entry["url"],
            "priority": priority,
            "source_type": source_type
        })
        
    # Sort by score descending
    scored_entries.sort(key=lambda x: x["score"], reverse=True)

    results = []
    sources = []
    seen_urls = set()
    seen_texts = set()

    print("\n--- DEBUG: RETRIEVAL RESULTS ---")
    for item in scored_entries:
        print(f"Score: {item['score']:.4f} | Priority: {item['priority']} | Type: {item['source_type']} | URL: {item['url']}")
        
        # Limit to top 5 chunks
        if len(results) >= 4:
            continue
            
        # Remove duplicate text chunks
        if item["text"] in seen_texts:
            continue
            
        results.append(item["text"])
        seen_texts.add(item["text"])

        if item["url"] not in seen_urls:
            sources.append(item["url"])
            seen_urls.add(item["url"])
    print("--------------------------------")

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

    prompt = f"""Provide a clear, concise, and professional answer using ONLY the provided context.
If the answer is not present in the context, respond with 'I don't know'.

Context:
{context}

Question:
{query}

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
