import ollama
from core.tools import search_docs, get_history_summary, reset_conversation
from core.memory import init_db, save_message
from logger import get_logger


logger = get_logger(__name__)



SYSTEM_PROMPT: str = """You are a helpful Python documentation assistant.
You answer questions strictly based on the provided documentation context.
If the context does not contain enough information, say so clearly.
Always be concise and precise. Use code examples where helpful.
Always answer in the language the user speaks to you! (code examples are excluded -> always english)."""



def handle_reset(user_input: str) -> str | None:
    """Use LLM to detect if the user wants to reset the conversation."""
    result = ollama.chat(
        model="qwen2.5:14b",
        messages=[
            {
                "role": "system",
                "content": "You are an intent classifier. Reply with only 'yes' or 'no'. Does the user want to reset, clear, or start a new conversation?"
            },
            {"role": "user", "content": user_input}
        ],
    )
    intent: str = result["message"]["content"].strip().lower()
    logger.debug(f"Reset intent: {intent}")
    if intent == "yes":
        return reset_conversation()
    return None




def build_message(user_input: str, context: str) -> list[dict]:
    """Build the full message list for the LLM including history and context."""
    history: list[dict] = get_history_summary()

    user_message: str = f"""Answer the following question using the documentation context below.

    Context:
    {context}

    Question: {user_input}"""

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    return messages



def run_agent(user_input: str) -> str:
    """Main agent function — takes user input and returns a response."""
    init_db()
    logger.info(f"User input: {user_input}")

    #check for reset intent
    reset_response = handle_reset(user_input)
    if reset_response:
        return reset_response
    

    #search docs
    context: str = search_docs(user_input)


    #handle escalation
    if context == "ESCALATE":
        response = (
            "I couldn't find relevant information in the Python documentation "
            "for your question. Please try rephrasing, or consult "
            "https://docs.python.org directly."
        )

        save_message("user", user_input)
        save_message("assistant", response)
        return response
    

    messages: list[dict] = build_message(user_input, context)


    result = ollama.chat(
        model="qwen2.5:14b",
        messages=messages,
    )

    response: str = result["message"]["content"]


    #save to memory
    save_message("user", user_input)
    save_message("assistant", response)

    logger.info("Response generated successfully.")
    return response



