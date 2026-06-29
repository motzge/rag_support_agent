import sqlite3
from datetime import datetime
from pathlib import Path
from logger import get_logger
from config import MEMORY_DB_PATH, MAX_HISTORY



logger = get_logger(__name__)



def init_db() -> None:
    """Create the memory database and table if they don't exist."""
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     role TEXT NOT NULL,
                     content TEXT NOT NULL,
                     timestamp TEXT NOT NULL
            )
        """)

        conn.commit()
    logger.debug("Memory database initialized.")




def save_message(role: str, content: str) -> None:
    """Save a single message to the conversation history."""
    timestamp = datetime.now().isoformat()
    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO conversation (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, timestamp),
        )

        #Delete oldest messages if limit is exceeded
        conn.execute("""
            DELETE FROM conversation
            WHERE id NOT IN(
                SELECT id FROM conversation
                ORDER BY id DESC LIMIT ?
                )
        """, (MAX_HISTORY,))
        conn.commit()
    logger.debug(f"saved message - role: {role}")




def load_history(limit: int = 10) -> list[dict]:
    """Load the last N messages from the conversation history."""
    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT role, content FROM conversation ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()


    #reverse so oldest message comes first
    history = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    logger.debug(f"loaded {len(history)} messages from memory.")
    return history 



def clear_history() -> None:
    """Delete all messages from the conversation history."""
    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        conn.execute("DELETE FROM conversation")
        conn.commit()
    logger.info("Conversation history cleaned.")



