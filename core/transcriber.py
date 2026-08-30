import os
import time
from pydub import AudioSegment
from groq import Groq
import groq as groq_module
from langsmith import traceable

# Groq has a 25MB file size limit for audio uploads.
# A 16kHz mono WAV is about 32KB/sec, so 25MB is ~13 minutes.
# We'll split the chunks into 4-minute pieces to be safe.
GROQ_PIECE_MINUTES = 4

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment or .env file.")
    return Groq(api_key=api_key)

@traceable(
    name="Groq Audio Translation",
    run_type="llm",
    tags=["stt", "groq", "translation"]
)
def _translate_piece(piece_path: str, client: Groq) -> str:
    """Send one piece to Groq whisper-large-v3 translation endpoint."""
    max_retries = 3
    base_delay = 2

    # Add file size to LangSmith metadata
    file_size_bytes = os.path.getsize(piece_path)
    
    for attempt in range(max_retries):
        try:
            with open(piece_path, "rb") as file:
                translation = client.audio.translations.create(
                    file=(os.path.basename(piece_path), file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    temperature=0.0
                )
            return translation.text
            
        except groq_module.AuthenticationError as e:
            # Bad API key — fail immediately, no point retrying
            raise RuntimeError("Groq authentication failed. Check your GROQ_API_KEY.") from e

        except (groq_module.APIConnectionError, groq_module.APITimeoutError) as e:
            # Network/connection/timeout errors — always retry
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  → Groq connection error. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Groq connection failed after {max_retries} attempts: {e}") from e

        except groq_module.RateLimitError as e:
            # Rate limit 429 — retry with backoff
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  → Groq rate limit hit. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Groq rate limit exceeded after {max_retries} attempts.") from e

        except groq_module.APIStatusError as e:
            # 5xx server errors — retry; 4xx client errors — fail fast
            if e.status_code >= 500 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  → Groq server error {e.status_code}. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Groq API error (HTTP {e.status_code}): {e.message}") from e

def translate_chunk_groq(chunk_path: str, client: Groq) -> str:
    """
    Split the chunk if needed, send to Groq, and return the English text.
    """
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = GROQ_PIECE_MINUTES * 60 * 1000
    
    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        # Ensure 16kHz mono to keep file size under Groq's 25MB limit
        piece = piece.set_channels(1).set_frame_rate(16000)
        piece_path = f"{chunk_path}_groq_{i}.wav"
        piece.export(piece_path, format="wav")
        
        try:
            print(f"  → Groq piece {i + 1}/{total_pieces} ...")
            # The _translate_piece is wrapped with @traceable
            text = _translate_piece(piece_path, client)
            full_text += text + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)
                
    return full_text.strip()

@traceable(name="Audio Transcription Pipeline", run_type="chain")
def transcribe_all(chunks: list) -> str:
    """
    Transcribe and translate all audio chunks using Groq.
    Always returns an English transcript.
    """
    client = get_groq_client()
    full_transcript = ""
    
    print("Using Groq (whisper-large-v3) for transcription and translation to English.")
    
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = translate_chunk_groq(chunk, client)
        full_transcript += text + " "
        
    print("Transcription complete.")
    return full_transcript.strip()
