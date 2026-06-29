from pathlib import Path
from dotenv import load_dotenv
import os



load_dotenv()



#Paths
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data" / "docs"
CHROMA_DIR: Path = BASE_DIR / "data" / "chroma"



#Ollama
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")



#RAG
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
TOP_K_RESULTS: int = 5



#Memory
MEMORY_DB_PATH: Path = BASE_DIR / "data" / "memory.db"
MAX_HISTORY: int = 100



#Escalation
CONFIDENCE_THRESHOLD: float = 0.8

