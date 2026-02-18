import faiss
import pickle
import psycopg2
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

DB_NAME = "gndec_rag"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_FILE = "faiss_index.bin"
ID_MAP_FILE = "id_map.pkl"

# =========================
# CONNECT DB
# =========================

print("Connecting to DB...")
conn = psycopg2.connect(dbname=DB_NAME)
cur = conn.cursor()

cur.execute("""
    SELECT url, section_title, content
    FROM pages
    WHERE content IS NOT NULL
""")

rows = cur.fetchall()
print(f"Fetched {len(rows)} rows from DB")

cur.close()
conn.close()

# =========================
# CHUNKING FUNCTION
# =========================

def chunk_text(text, chunk_size=700, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# =========================
# PREPARE TEXTS
# =========================

texts = []
metadata = []

for url, section_title, content in rows:
    full_text = f"{section_title}\n{content}"

    chunks = chunk_text(full_text)

    for chunk in chunks:
        if len(chunk.strip()) > 50:
            texts.append(chunk)
            metadata.append(url)

print(f"Total chunks created: {len(texts)}")

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# =========================
# BUILD FAISS INDEX
# =========================

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(embeddings)

print("Saving index...")
faiss.write_index(index, INDEX_FILE)

with open(ID_MAP_FILE, "wb") as f:
    pickle.dump(metadata, f)

print("Index build complete.")
