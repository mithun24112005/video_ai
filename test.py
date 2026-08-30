import os
import sys
from dotenv import load_dotenv

# Test observability loading
load_dotenv()
if os.getenv("LANGSMITH_TRACING") == "true":
    print("[OK] LangSmith tracing is enabled via environment.")
else:
    print("[WARN] LangSmith tracing is NOT enabled.")

from utils.audio_processor import process_input
from core.transcriber import transcribe_all, get_groq_client
from core.vector_store import get_qdrant_client, HFInferenceEmbeddings, build_vector_store
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

def run_tests():
    print("\n" + "=" * 60)
    print("[RUN] RUNNING SYSTEM TESTS")
    print("=" * 60)

    # 1. Test HF Embeddings Initialization
    print("\n[1] Testing HF Embeddings Initialization...")
    try:
        embeddings = HFInferenceEmbeddings()
        print("[OK] HFInferenceEmbeddings initialized successfully.")
        
        # Test shape
        sample_emb = embeddings.embed_query("test query")
        print(f"[OK] Embedding query successful. Dimension: {len(sample_emb)}")
    except Exception as e:
        print(f"[FAIL] HF Embeddings failed: {e}")
        return

    # 2. Test Qdrant Connection
    print("\n[2] Testing Qdrant Connection...")
    try:
        q_client = get_qdrant_client()
        collections = q_client.get_collections()
        print(f"[OK] Qdrant connected. Collections available: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"[FAIL] Qdrant connection failed: {e}")
        print("   Make sure Docker container is running: docker compose up -d")
        return
        
    # 3. Test Groq Client Initialization
    print("\n[3] Testing Groq Client Initialization...")
    try:
        g_client = get_groq_client()
        print("[OK] Groq client initialized successfully.")
    except Exception as e:
        print(f"[FAIL] Groq client failed: {e}")
        return

    print("\n[OK] All basic infrastructure tests passed.")
    print("\nTo test the full pipeline, run: uv run python main.py")

if __name__ == "__main__":
    run_tests()