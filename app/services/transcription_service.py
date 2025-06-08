import io
import requests
from requests.auth import HTTPBasicAuth
from pydub import AudioSegment
from faster_whisper import WhisperModel
import tempfile # Needed if you still wanted to use temporary files, but we'll avoid it for in-memory processing
import numpy as np # Often useful if converting audio to numpy arrays for models

from app.core.config import settings

# Initialize Faster-Whisper model globally once
# This model will be reused for all transcription requests
model_size = "medium"  # "tiny", "base", "small", "medium", "large-v3", "distil-large-v3"
WHISPER_MODEL = None # Initialize to None in case of loading error
try:
    WHISPER_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8") # Assuming CPU usage
    print(f"🧠 Faster-Whisper model '{model_size}' loaded successfully on CPU.")
except Exception as e:
    print(f"🚨 Error loading Faster-Whisper model: {e}")
    print("Please ensure the model files are downloaded and environment is set up correctly.")


def transcribe_audio_from_url(media_url: str) -> str:
    """
    Downloads an audio file from Twilio and transcribes it using Faster-Whisper.
    """
    if WHISPER_MODEL is None:
        return "Error: Transcription model is not loaded. Cannot process audio from URL."

    try:
        print(f"🔐 Downloading audio from: {media_url}")

        response = requests.get(
            media_url,
            auth=HTTPBasicAuth(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        )
        if response.status_code != 200:
            print(f"❌ Failed to download audio. Status: {response.status_code}")
            return ""

        audio_data_io = io.BytesIO(response.content)
        
        # Use pydub to process the audio, then export to WAV in-memory for Faster-Whisper
        audio = AudioSegment.from_file(audio_data_io)
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0) # Rewind the buffer to the beginning

        print("🧠 Transcribing with Faster-Whisper from URL...")
        segments, info = WHISPER_MODEL.transcribe(
            wav_buffer, # Faster-Whisper can often take a file-like object directly
            beam_size=7,
            vad_filter=True,
            word_timestamps=False,
            initial_prompt="Trading, finance, market signals, gold, stocks, forex."
        )

        transcription = " ".join([segment.text for segment in segments])
        print(f"✅ Transcription successful: '{transcription.strip()}'")
        return transcription.strip()

    except Exception as e:
        print(f"🚨 Transcription error from URL: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return ""


def transcribe_audio_from_bytes(audio_bytes: bytes) -> str:
    """
    Transcribes raw audio bytes (e.g., from browser's MediaRecorder, usually WebM format).
    """
    if WHISPER_MODEL is None:
        return "Error: Transcription model is not loaded. Cannot process audio from bytes."

    try:
        print(f"Transcribing audio from raw bytes (assuming WebM format)...")
        
        # Load audio bytes into pydub, specifying the format (client sends webm)
        audio_data_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_data_io, format="webm")
        
        # Export as WAV in-memory to be consumed by Faster-Whisper
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0) # Rewind the buffer to the beginning

        segments, info = WHISPER_MODEL.transcribe(
            wav_buffer, # Pass the BytesIO object directly
            beam_size=5,
            vad_filter=True,
            word_timestamps=False
        )

        transcribed_text = " ".join([segment.text for segment in segments])
        print(f"✅ Transcription successful from bytes: '{transcribed_text.strip()}'")
        return transcribed_text.strip()

    except Exception as e:
        print(f"🚨 Error during direct audio transcription from bytes: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return "Error processing audio."