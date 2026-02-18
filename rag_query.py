import faiss
import pickle
import psycopg2
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

DB_NAME = "gndec_rag"
INDEX_FILE = "faiss_index.bin"
ID_MAP_FILE = "id_map.pkl"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
TOP_K = 8
MAX_CONTEXT_CHARS = 9000

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

# =========================
# LOAD FAISS
# =========================

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

with open(ID_MAP_FILE, "rb") as f:
    id_map = pickle.load(f)

# =========================
# CONNECT DB
# =========================

conn = psycopg2.connect(dbname=DB_NAME)
cur = conn.cursor()

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
    sources = set()

    for idx in indices[0]:
        if idx == -1:
            continue

        url = id_map[idx]

        cur.execute("""
            SELECT section_title, content
            FROM pages
            WHERE url = %s
            LIMIT 1
        """, (url,))

        row = cur.fetchone()
        if row:
            section_title, content = row
            results.append(f"{section_title}\n{content}")
            sources.add(url)

    return results, list(sources)

# =========================
# MAIN LOOP
# =========================

while True:
    query = input("\nAsk: ")

    if query.lower() in ["exit", "quit"]:
        break

    print("Retrieving context...")

    retrieved_chunks, sources = retrieve(query)

    if not retrieved_chunks:
        print("\nAnswer:\nNo relevant information found in GNDEC data.")
        continue

    context = "\n\n".join(retrieved_chunks)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are a helpful assistant answering questions about GNDEC.
Answer strictly using the provided context.
If answer not found, say "Information not found in GNDEC data."

Context:
{context}

Question:
{query}
"""

    print("Sending to LLM...")

    response = requests.post(
        LLAMA_URL,
        json={
            "model": "Llama-3.2-3B-Instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
    )

    data = response.json()

    if "choices" not in data:
        print("LLM Error:", data)
        continue

    answer = data["choices"][0]["message"]["content"]

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:")
    for src in sources:
        print("-", src)

cur.close()
conn.close()
