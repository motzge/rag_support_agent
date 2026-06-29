import ollama
from core.tools import search_docs, get_history_summary, reset_conversation
from core.memory import init_db, save_message
from logger import get_logger


logger = get_logger(__name__)



SYSTEM_PROMPT: str = """You are a Python documentation assistant.

Answer strictly and only from the documentation context provided below the question.
If that context does not contain the answer, say clearly that it is not covered by the
documentation — and stop there. Do not answer from your own knowledge, do not guess,
and do not soften this with words like "typically", "usually", or "however, you could".
Never add examples or code for anything that is not in the provided context.

Be concise and precise. Lead with a direct one-sentence answer, then add detail
only if it genuinely helps. Prefer plain language over exhaustive technical
edge-cases. Use code examples from the context where they help.
Always answer in the language the user speaks to you (code examples stay English)."""



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



def rewrite_query(user_input: str, history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone search query using history."""
    # first turn has no context to resolve against — search as-is
    if not history:
        return user_input

    # only the last few turns matter for resolving references like "it" or "an example"
    recent: str = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])

    result = ollama.chat(
        model="qwen2.5:14b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's follow-up into a standalone search query. "
                    "Resolve references (it, that, an example) using the conversation. "
                    "Output ONLY the rewritten query, nothing else. "
                    "If it is already standalone, return it unchanged."
                ),
            },
            {"role": "user", "content": f"Conversation:\n{recent}\n\nFollow-up: {user_input}"},
        ],
    )
    rewritten: str = result["message"]["content"].strip()
    logger.debug(f"Query rewrite: {user_input!r} -> {rewritten!r}")
    return rewritten



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
    #rewrite follow-ups into standalone queries, then search
    history: list[dict] = get_history_summary()
    search_query: str = rewrite_query(user_input, history)
    context: str = search_docs(search_query)


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



