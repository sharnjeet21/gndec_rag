import faiss
import pickle
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

DATA_FILE = "data_final.json"
INDEX_FILE = "faiss_index.bin"
ID_MAP_FILE = "id_map.pkl"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

with open(DATA_FILE, "r") as f:
    rows = json.load(f)

print(f"Loaded {len(rows)} records")

# =========================
# CHUNKING FUNCTION
# =========================

def chunk_text(text, chunk_size=700, overlap=150):
    chunks = []
    text = str(text)

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# =========================
# PREPARE TEXTS (CLEAN)
# =========================

texts = []
metadata = []
seen_chunks = set()

print("Preparing chunks...")

for item in rows:
    try:
        url = str(item.get("url", ""))

        section_title = str(item.get("section_title", ""))
        content = str(item.get("content", ""))

        full_text = section_title + "\n" + content

        if len(full_text.strip()) == 0:
            continue

        chunks = chunk_text(full_text)

        for chunk in chunks:
            if not isinstance(chunk, str):
                continue

            chunk = chunk.strip()

            if len(chunk) < 50:
                continue

            # UTF-8 safe
            try:
                chunk.encode("utf-8")
            except:
                continue

            if chunk in seen_chunks:
                continue

            seen_chunks.add(chunk)

            texts.append(chunk)
            metadata.append({
                "url": url,
                "text": chunk
            })

    except Exception as e:
        print("Skipping bad record:", e)

print(f"Total clean chunks: {len(texts)}")

# =========================
# FINAL CLEAN FILTER
# =========================

texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]

print("Final texts count:", len(texts))

# =========================
# LOAD MODEL (GPU)
# =========================

print("Loading embedding model on GPU...")

model = SentenceTransformer(EMBED_MODEL)

# =========================
# GENERATE EMBEDDINGS
# =========================

print("Generating embeddings...")

try:
    embeddings = model.encode(
        texts,
        batch_size=32,   # GPU optimized
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

except Exception as e:
    print("Batch encoding failed:", e)
    print("Falling back to safe encoding...")

    embeddings_list = []
    for t in tqdm(texts):
        try:
            emb = model.encode(
                t,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings_list.append(emb)
        except:
            continue

    embeddings = np.array(embeddings_list)

# =========================
# BUILD FAISS INDEX
# =========================

dimension = embeddings.shape[1]

print("Building FAISS index...")

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# =========================
# SAVE
# =========================

print("Saving index...")

faiss.write_index(index, INDEX_FILE)

with open(ID_MAP_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"Index build complete: {len(embeddings)} vectors stored.")