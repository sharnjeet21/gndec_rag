
# GNDEC RAG Advanced

This project scrapes GNDEC websites, extracts text from HTML, PDF, and DOCX files, stores it in PostgreSQL, builds a FAISS index with embeddings, and allows retrieval-augmented generation (RAG) queries using a local LLM.

## Features

- Web crawler for GNDEC domains (`gndec.ac.in`)  
- Extracts content from:
  - HTML pages
  - PDF files
  - DOCX files  
- Stores data in PostgreSQL  
- Builds FAISS index with sentence-transformers embeddings  
- Query interface using local LLM  

## Requirements

- Python 3.12+  
- PostgreSQL 14+  
- Git  
- C++ compiler for llama.cpp  
- `pip` packages:

```bash
pip install -r requirements.txt


Required Python packages:

psycopg2

requests

beautifulsoup4

pypdf

python-docx

sentence-transformers

faiss-cpu or faiss-gpu

Setup
1. Clone Repository
git clone https://github.com/yourusername/gndec-rag-advanced.git
cd gndec-rag-advanced

2. Create Python Virtual Environment
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install --upgrade pip
pip install -r requirements.txt

3. Configure PostgreSQL

Create a database:

createdb gndec_rag


Update your Python scripts if your DB username/password is not default.

4. Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make


On Linux, you can add -DUSE_CUBLAS=ON in the Makefile or make command if you want GPU support with cuBLAS.

5. Download LLaMA Model

Copy or download your quantized LLaMA model to a folder:

hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF


Make sure your llama-server command points to the correct path:

./llama-server -hf hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF:Q4_K_M -ngl 999 -c 10240 -t 8


This starts the local LLM server (GPU if compiled with GPU support).

6. Run Crawler
python crawler.py


The crawler skips images, audio, video, and /gallery URLs. PDFs and DOCX files store content and URLs.

7. Build FAISS Index
python build_index.py

8. Query RAG
python rag_query.py


Type your question when prompted. The script fetches context from FAISS and sends it to your local LLM.

Notes

Ensure your LLM server is running before running rag_query.py.

FAISS index and DB must be in sync; otherwise, some queries may return incomplete results.

You can adjust crawling speed, MAX_FILE_SIZE, and ignored extensions in crawler.py.

On Linux/Manjaro, GPU acceleration requires llama.cpp compiled with USE_CUBLAS=ON and CUDA installed.

Project Structure
gndec-rag-advanced/
│
├─ crawler.py          # Web crawler
├─ build_index.py      # Build embeddings and FAISS index
├─ rag_query.py        # RAG query interface
├─ faiss_index.bin     # Generated FAISS index (after running)
├─ id_map.pkl          # Mapping of DB IDs to index
├─ temp_files/         # Temporary files for PDFs/DOCX
├─ requirements.txt
├─ README.md
└─ venv/               # Python virtual environment

Quick Start (3 Commands)
# Activate environment
source venv/bin/activate

# Run crawler
python crawler.py

# Build index
python build_index.py

# Query
python rag_query.py

License

MIT License


---

If you want, I can also **write a ready-to-use `requirements.txt`** that includes **GPU-compatible FAISS and all necessary packages**, so anyone can just do `pip install -r requirements.txt` and start.  

Do you want me to create that file next?