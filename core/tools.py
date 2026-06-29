from core.retriever import retrieve, RetrievalResult
from core.memory import load_history, clear_history
from logger import get_logger
from config import CONFIDENCE_THRESHOLD


logger = get_logger(__name__)



def search_docs(query: str) -> str:
    """Search the Python docs and return formatted context for the LLM."""
    logger.debug(f"Tool called: search_docs | query: {query}")
    results: list[RetrievalResult] = retrieve(query)

    if not results:
        return "No relevant documentation found."
    
    #check if best result is confident enough
    best_distance: float = results[0].distance
    if best_distance > CONFIDENCE_THRESHOLD:
        logger.info(f"low confidence - best distance: {best_distance:.4f}")
        return "ESCALATE"
    

    #format chunks as conterxt block for the LLM
    context_parts: list[str] = []
    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"[Source {i}: {result.source}]\n{result.content}"
        )

    return "\n\n".join(context_parts)



def get_history_summary() -> list[dict]:
    """Return the current conversation history for the LLM."""
    return load_history()



def reset_conversation() -> str:
    """Clear the conversation history and confirm to the user."""
    clear_history()
    logger.info("Conversation reset by user.")
    return "Conversation history cleared. Starting fresh!"



    
