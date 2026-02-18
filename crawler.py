import os
import time
import requests
import psycopg2
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pypdf import PdfReader
from docx import Document

# =========================
# CONFIG
# =========================

BASE_URL = "https://gndec.ac.in"
SKIP_PATHS = ["/gallery"]

DOWNLOAD_DIR = "temp_files"
MAX_FILE_SIZE = 10_000_000  # 10MB limit

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

visited = set()
queue = [BASE_URL]

# =========================
# MEDIA EXTENSIONS TO IGNORE
# =========================

MEDIA_EXTENSIONS = {
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg",
    ".tiff", ".tif", ".ico", ".heic", ".heif",

    # Video
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".wmv", ".flv",
    ".webm", ".mpg", ".mpeg", ".3gp", ".3g2", ".ts", ".mts",

    # Audio
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".oga",
    ".wma", ".m4a", ".amr", ".opus",

    # Archives
    ".zip", ".rar", ".7z", ".tar", ".gz",

    # Executables
    ".exe", ".msi", ".dmg", ".apk"
}

# =========================
# DATABASE CONNECTION
# =========================

conn = psycopg2.connect(dbname="gndec_rag")
cur = conn.cursor()

# =========================
# HELPERS
# =========================

def should_skip(url):
    for path in SKIP_PATHS:
        if path in url:
            return True
    return False


def is_internal(url):
    parsed = urlparse(url)
    return parsed.netloc.endswith("gndec.ac.in")


def clean_text(soup):
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup


def extract_sections(soup):
    sections = []
    headers = soup.find_all(["h1", "h2", "h3"])

    for header in headers:
        section_title = header.get_text(strip=True)
        content = []

        for sibling in header.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            content.append(sibling.get_text(" ", strip=True))

        full_text = " ".join(content)

        if len(full_text) > 100:
            sections.append((section_title, full_text))

    return sections


def save_to_db(url, title, section_title, content, file_type="html"):
    try:
        cur.execute("""
            INSERT INTO pages (url, title, section_title, content, file_type)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (url, title, section_title, content, file_type))
        conn.commit()
    except Exception as e:
        print("DB error:", e)


def safe_head_request(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.headers
    except:
        return {}

# =========================
# PDF EXTRACTION
# =========================

def extract_pdf_text(url):
    try:
        headers = safe_head_request(url)
        size = int(headers.get("Content-Length", 0))

        if size > MAX_FILE_SIZE:
            print("Large PDF skipped:", url)
            return ""

        filename = os.path.join(DOWNLOAD_DIR, "temp.pdf")
        response = requests.get(url, timeout=20)

        with open(filename, "wb") as f:
            f.write(response.content)

        reader = PdfReader(filename)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        os.remove(filename)
        return text.strip()

    except Exception as e:
        print("PDF error:", e)
        return ""

# =========================
# DOCX EXTRACTION
# =========================

def extract_docx_text(url):
    try:
        headers = safe_head_request(url)
        size = int(headers.get("Content-Length", 0))

        if size > MAX_FILE_SIZE:
            print("Large DOCX skipped:", url)
            return ""

        filename = os.path.join(DOWNLOAD_DIR, "temp.docx")
        response = requests.get(url, timeout=20)

        with open(filename, "wb") as f:
            f.write(response.content)

        doc = Document(filename)
        text = "\n".join([para.text for para in doc.paragraphs])

        os.remove(filename)
        return text.strip()

    except Exception as e:
        print("DOCX error:", e)
        return ""

# =========================
# MAIN CRAWLER LOOP
# =========================

while queue:
    url = queue.pop(0)

    if url in visited:
        continue

    if should_skip(url):
        print("Skipping restricted path:", url)
        continue

    parsed_path = urlparse(url).path.lower()

    # Skip all media/archives/executables
    if any(parsed_path.endswith(ext) for ext in MEDIA_EXTENSIONS):
        print("Media skipped:", url)
        continue

    print("Scraping:", url)
    visited.add(url)

    try:
        # -------- PDF --------
        if parsed_path.endswith(".pdf"):
            pdf_text = extract_pdf_text(url)

            if len(pdf_text) > 200:
                save_to_db(url, "PDF Document", "PDF Content", pdf_text, "pdf")
                save_to_db(url, "PDF Document", "Source URL", url, "pdf_url")

            continue

        # -------- DOCX --------
        if parsed_path.endswith(".docx"):
            doc_text = extract_docx_text(url)

            if len(doc_text) > 200:
                save_to_db(url, "DOCX Document", "DOCX Content", doc_text, "docx")
                save_to_db(url, "DOCX Document", "Source URL", url, "docx_url")

            continue

        # -------- HTML --------
        response = requests.get(url, timeout=10)

        if "text/html" not in response.headers.get("Content-Type", ""):
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        soup = clean_text(soup)

        title = soup.title.string.strip() if soup.title else ""

        sections = extract_sections(soup)

        for section_title, content in sections:
            save_to_db(url, title, section_title, content, "html")

        # Extract internal links
        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])
            full_url = full_url.split("#")[0]

            if not full_url.startswith("http"):
                continue

            if is_internal(full_url) and full_url not in visited:
                queue.append(full_url)

        time.sleep(1)

    except Exception as e:
        print("Error:", e)

cur.close()
conn.close()

print("Crawling complete.")
