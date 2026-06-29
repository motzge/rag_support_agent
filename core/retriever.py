from dataclasses import dataclass
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from logger import get_logger
from config import CHROMA_DIR, OLLAMA_EMBED_MODEL, TOP_K_RESULTS



logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its metadata."""
    content: str
    source: str
    chunk_index: str
    distance: float



def get_collection() -> chromadb.Collection:
    """Open the existing ChromaDB collection."""
    embed_fn = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=OLLAMA_EMBED_MODEL,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(
        name="python_docs",
        embedding_function=embed_fn,
    )



def retrieve(query: str) -> list[RetrievalResult]:
    """Search ChromaDB for the most relevant chunks for a given query."""
    logger.debug(f"Retrieving Chunks for query: {query}")

    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K_RESULTS,
    )

    chunks: list[RetrievalResult] = []


    documents: list[str] = results["documents"][0]
    metadatas: list[dict] = results["metadatas"][0]
    distances: list[float] = results["distances"][0]


    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(RetrievalResult(
            content=doc,
            source=meta.get("source", "unknown"),
            chunk_index=meta.get("chunk_index", 0),
            distance=dist,
        ))


    logger.debug(f"Found {len(chunks)} chunks, best distance: {distances[0]:.4f}")
    return chunks 