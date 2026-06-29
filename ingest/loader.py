import time
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from logger import get_logger
from config import (
    DATA_DIR,
    CHROMA_DIR,
    OLLAMA_EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


logger = get_logger(__name__)

PYTHON_DOCS_SITEMAP: str = "https://docs.python.org/3/genindex-all.html"




def fetch_page(url: str) -> str:
    """Download HTML of a page and extract plain text."""

    logger.debug(f"fetching: {url}")
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    #only the main content - Navigation, Header, Footer gets cut-out
    main = soup.find("div", {"role": "main"}) or soup.find("body")
    return main.get_text(separator=" ", strip=True)




def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""

    chunks: list[str] = []
    start: int = 0

    while start < len(text):
        end: int = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks




def fetch_all_urls() -> list[str]:
    """Fetch all documentation URLs from the Python docs index."""
    logger.info("Fetching all URLs from Python docs index...")
    response = requests.get(PYTHON_DOCS_SITEMAP, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    base_url = "https://docs.python.org/3/"
    urls: list[str] = []

    for a_tag in soup.find_all("a", href=True):
        href: str = a_tag["href"]
        if href.startswith("#") or href.startswith("http"):
            continue
        full_url = base_url + href.split("#")[0]
        if full_url not in urls:
            urls.append(full_url)

    logger.info(f"Found {len(urls)} unique URLs.")
    return urls




def get_chroma_collection() -> chromadb.Collection:
    """Initialize ChromaDB client and collection."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embed_fn = OllamaEmbeddingFunction(                #Pylance doesnt know the ChromaDb Embedding Function Signatur... should still work! 
        url = "http://localhost:11434/api/embeddings",
        model_name = OLLAMA_EMBED_MODEL,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="python_docs",
        embedding_function=embed_fn,
    )

    return collection 




def ingest_docs() -> None:
    """Main function — load all pages, chunk, and store in ChromaDB."""
    collection = get_chroma_collection()

    existing: int = collection.count()
    if existing > 0:
        logger.info(f"Collection already has {existing} chunks — skipping ingest.")
        return

    urls: list[str] = fetch_all_urls()
    logger.info(f"Starting ingest for {len(urls)} pages...")
    total_chunks: int = 0

    for url in urls:
        try:
            text: str = fetch_page(url)
            chunks: list[str] = chunk_text(text)

            ids: list[str] = [f"{url}#{i}" for i in range(len(chunks))]
            metadatas: list[dict] = [{"source": url, "chunk_index": i} for i in range(len(chunks))]

            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)
            logger.info(f"ok {url} -> {len(chunks)} chunks")
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"Error on {url}: {e}")
            continue

    logger.info(f"Ingest done — {total_chunks} chunks total in ChromaDB.")




def extract_text_from_upload(file) -> str:
    """Read text from an uploaded file (txt, md, or pdf)."""
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file)
        pages: list[str] = [page.extract_text() or "" for page in reader.pages]
        return " ".join(pages)
    return file.read().decode("utf-8", errors="ignore")




def ingest_file(file) -> int:
    """Chunk a user-uploaded file and store it in ChromaDB. Returns chunk count."""
    collection = get_chroma_collection()
    text: str = extract_text_from_upload(file)
    if not text.strip():
        logger.warning(f"No text extracted from {file.name}")
        return 0
    chunks: list[str] = chunk_text(text)
    ids: list[str] = [f"upload::{file.name}#{i}" for i in range(len(chunks))]
    metadatas: list[dict] = [{"source": file.name, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info(f"Ingested upload {file.name} -> {len(chunks)} chunks")
    return len(chunks)





if __name__ == "__main__":
    ingest_docs()