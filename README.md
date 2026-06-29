# Python Docs Assistant

A local-first RAG chatbot that answers questions about Python using the
official documentation as its knowledge base. Runs entirely offline through
Ollama — no API keys, no data leaving your machine.

## What it does

- Retrieves relevant passages from the indexed Python docs (vector search)
- Answers questions grounded in those passages, with conversation memory
- Lets you drop in your own `.pdf` / `.txt` / `.md` files via the UI to
  extend the knowledge base on the fly

## Stack

- **LLM:** Ollama (`qwen2.5:14b`) — local
- **Embeddings:** Ollama (`nomic-embed-text`) — local
- **Vector DB:** ChromaDB
- **Memory:** SQLite
- **UI:** Streamlit

## Project structure

\`\`\`
.
├── app.py              # Streamlit entry point
├── config.py           # Settings (loaded from .env)
├── logger.py           # Central logging
├── main.py             # CLI mode - same agent, no UI (interactive loop)
├── core/
│   ├── agent.py        # Orchestration: retrieve -> generate
│   ├── retriever.py    # Vector search over ChromaDB
│   ├── memory.py       # Conversation history (SQLite)
│   └── tools.py        # Agent tools
├── ingest/
│   └── loader.py       # Scrapes & chunks docs, builds the vector store
└── data/               # Vector store, memory db, logs (gitignored)
\`\`\`

## Setup

1. **Install [Ollama](https://ollama.com)** and pull the models:
   \`\`\`bash
   ollama pull qwen2.5:14b
   ollama pull nomic-embed-text
   \`\`\`

2. **Create a virtual environment and install dependencies:**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   \`\`\`

3. **Configure environment:**
   \`\`\`bash
   cp .env.example .env
   \`\`\`

4. **Build the knowledge base** (scrapes & indexes the Python docs from https://docs.python.org/3/):
   \`\`\`bash
   python ingest/loader.py        
   \`\`\`

5. **Run:**
   \`\`\`bash
   streamlit run app.py     # web UI
   python main.py           # or: CLI mode
   \`\`\`

## Notes

The vector store and conversation database are generated locally and are not
tracked in git. Step 4 rebuilds them from scratch.