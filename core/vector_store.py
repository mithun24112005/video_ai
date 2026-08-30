import os
import uuid
from typing import List
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langsmith import traceable

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "meeting_transcript")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_EMBEDDING_BATCH_SIZE = int(os.getenv("HF_EMBEDDING_BATCH_SIZE", "32"))


class HFInferenceEmbeddings(Embeddings):
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set in environment or .env file.")
        self.client = InferenceClient(token=token)
        self.model = HF_EMBEDDING_MODEL
        self.batch_size = HF_EMBEDDING_BATCH_SIZE

    @traceable(name="HuggingFace Feature Extraction (Batch)", run_type="embedding", tags=["huggingface", "embedding"])
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                # The Inference API returns a numpy array or nested list of embeddings
                response = self.client.feature_extraction(text=batch, model=self.model)
                # Feature extraction returns List[List[float]] or a numpy array equivalent
                import numpy as np
                # Ensure it's a list of lists of floats
                batch_embeddings = np.array(response).tolist()
                
                # If a single dimension array was returned accidentally
                if len(batch) == 1 and not isinstance(batch_embeddings[0], list):
                    batch_embeddings = [batch_embeddings]
                    
                embeddings.extend(batch_embeddings)
            except Exception as e:
                raise RuntimeError(f"HuggingFace embedding failed for batch {i//self.batch_size}: {e}") from e
                
        if len(embeddings) != len(texts):
            raise RuntimeError("Number of embeddings returned does not match number of input texts.")
            
        return embeddings

    @traceable(name="HuggingFace Feature Extraction (Query)", run_type="embedding")
    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.feature_extraction(text=text, model=self.model)
            import numpy as np
            emb = np.array(response).tolist()
            # If batch dim is present [1, dim]
            if len(emb) == 1 and isinstance(emb[0], list):
                return emb[0]
            return emb
        except Exception as e:
            raise RuntimeError(f"HuggingFace query embedding failed: {e}") from e


def get_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(url=QDRANT_URL)
        # Check connection
        client.get_collections()
        return client
    except Exception as e:
        raise RuntimeError(f"Could not connect to Qdrant at {QDRANT_URL}. Ensure it is running (docker compose up -d). Error: {e}") from e

@traceable(name="Build Vector Store", run_type="chain")
def build_vector_store(transcript: str) -> QdrantVectorStore:
    print(f"Building vector Store in Qdrant ({QDRANT_URL})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    # We generate a unique document ID for this transcript run
    # to avoid mixing with previous transcripts. 
    # Or, simpler: we just recreate the collection completely.
    client = get_qdrant_client()
    embeddings = HFInferenceEmbeddings()

    # Determine embedding dimension from a single chunk
    sample_emb = embeddings.embed_query(chunks[0] if chunks else "test")
    dim = len(sample_emb)
    print(f"Detected embedding dimension: {dim}")

    # Recreate collection to prevent stale vector data
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i, 'source': 'meeting_transcript'})
        for i, chunk in enumerate(chunks)
    ]

    print(f"Inserting {len(docs)} chunks into Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
    
    # LangChain Qdrant integration
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )
    
    # Add documents (it uses client under the hood)
    vector_store.add_documents(docs)

    return vector_store


def load_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()
    embeddings = HFInferenceEmbeddings()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )
    return vector_store


def get_retriever(vector_store: QdrantVectorStore, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )
