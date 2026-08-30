# AI Video Assistant

An AI-powered meeting and video assistant that transcribes audio, translates it to English, summarizes, extracts action items, and lets you chat with the transcript using RAG (Retrieval-Augmented Generation).

## Architecture

This project is built using:
- **Groq (`whisper-large-v3`)**: Fast, multilingual speech-to-text and translation to English.
- **Mistral AI**: LLM for summarization, entity extraction, and RAG reasoning.
- **HuggingFace Inference API**: Remote embedding generation (no local models downloaded).
- **Qdrant**: Vector database running in Docker.
- **LangChain**: Pipeline orchestration and LCEL chains.
- **LangSmith**: Observability, tracing, and latency monitoring.
- **Streamlit**: Web interface.

## Prerequisites

1. **Python 3.13** or higher
2. **uv** (Fast Python package installer)
3. **Docker Compose** (for running Qdrant)

## Setup

1. **Clone and Setup Virtual Environment:**
   If you haven't already, ensure you're in the project directory. We use `uv` for dependency management:
   ```bash
   uv sync
   ```

2. **Start Qdrant Vector Database:**
   ```bash
   docker compose up -d
   ```
   *Qdrant runs locally on port `6333` and persists data in a Docker volume. Dashboard is at `http://localhost:6333/dashboard`*

3. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   # Required Keys
   GROQ_API_KEY=your_groq_key
   HF_TOKEN=your_huggingface_token
   MISTRAL_API_KEY=your_mistral_key

   # Qdrant Configuration
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION_NAME=meeting_transcript

   # Embeddings Configuration
   HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   HF_EMBEDDING_BATCH_SIZE=32
   ```

## LangSmith Observability

This project has native LangSmith observability to trace the execution pipeline, LLM calls, embeddings, and RAG retrievals.

To enable LangSmith tracing, add the following to your `.env`:
```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=ai-video-assistant
```

*Note: LangSmith is optional. The application will still function completely if `LANGSMITH_TRACING=false` or if no key is provided.*

## Usage

### 1. Web UI (Streamlit)
Launch the interactive web application:
```bash
uv run streamlit run app.py
```

### 2. Command Line Interface
Run the backend pipeline directly from the terminal:
```bash
uv run python main.py
```
*Enter a YouTube URL or a local video/audio file path when prompted.*

## Testing
Run the system integration test to verify infrastructure components (Qdrant, Groq, HF Embeddings):
```bash
uv run python test.py
```
