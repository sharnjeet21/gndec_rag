"""
Rebuilds id_map.pkl from the PostgreSQL DB using the same chunking logic
as build_index.py — without re-generating embeddings.

The FAISS index has 68,579 vectors; this script must produce exactly that
many entries in the same order so indices align correctly.
"""

import pickle
import psycopg2

DB_CONFIG = {
    "dbname": "gndec_rag",
    "user": "postgres",
    "password": "",
    "host": "localhost",
    "port": 5432,
}

ID_MAP_FILE = "id_map.pkl"
EXPECTED_VECTORS = 68579  # must match faiss_index.bin ntotal


def chunk_text(text, chunk_size=700, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


print("Connecting to DB...")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("""
    SELECT url, section_title, content
    FROM pages
    WHERE content IS NOT NULL AND content != ''
""")
rows = cur.fetchall()
cur.close()
conn.close()
print(f"Fetched {len(rows)} rows")

metadata = []
seen_chunks = set()

for url, section_title, content in rows:
    full_text = f"{section_title}\n{content}"
    chunks = chunk_text(full_text)
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 50 and chunk not in seen_chunks:
            seen_chunks.add(chunk)
            metadata.append({"url": url, "text": chunk})

print(f"Total unique chunks: {len(metadata)}")

if len(metadata) != EXPECTED_VECTORS:
    print(f"WARNING: chunk count ({len(metadata)}) != FAISS vectors ({EXPECTED_VECTORS})")
    print("The index and id_map may be misaligned. Consider running build_index.py to rebuild both.")
else:
    print("Chunk count matches FAISS index.")

with open(ID_MAP_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"Saved {ID_MAP_FILE} with {len(metadata)} entries.")
print("Sample entry:", metadata[0])
