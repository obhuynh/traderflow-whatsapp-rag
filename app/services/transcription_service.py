import io
import requests
from requests.auth import HTTPBasicAuth
from pydub import AudioSegment
from faster_whisper import WhisperModel
import numpy as np
import logging # Use standard logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Faster-Whisper model globally once
model_size = "medium"
WHISPER_MODEL = None
try:
    WHISPER_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
    logger.info(f"🧠 Faster-Whisper model '{model_size}' loaded successfully on CPU.")
except Exception as e:
    logger.exception(f"🚨 Error loading Faster-Whisper model: {e}")
    logger.error("Please ensure the model files are downloaded and environment is set up correctly.")


def transcribe_audio_from_url(media_url: str) -> str:
    """
    Downloads an audio file from Twilio and transcribes it using Faster-Whisper.
    """
    if WHISPER_MODEL is None:
        logger.error("Transcription model is not loaded for URL transcription.")
        return "Error: Transcription model is not loaded. Cannot process audio from URL."

    try:
        logger.info(f"🔐 Downloading audio from: {media_url}")

        response = requests.get(
            media_url,
            auth=HTTPBasicAuth(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        )
        if response.status_code != 200:
            logger.error(f"❌ Failed to download audio from URL {media_url}. Status: {response.status_code}")
            return ""

        audio_data_io = io.BytesIO(response.content)
        
        # Use pydub to process the audio, then export to WAV in-memory for Faster-Whisper
        # Added error handling for pydub's from_file and explicit format
        try:
            audio = AudioSegment.from_file(audio_data_io, format="webm") # Assume webm, browser common
        except Exception as pydub_e:
            logger.error(f"🚨 Pydub failed to decode audio from URL (assumed webm): {pydub_e}")
            return "Error decoding audio from URL."

        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav") # Export as WAV for Whisper
        wav_buffer.seek(0)

        logger.info("🧠 Transcribing with Faster-Whisper from URL...")
        segments, info = WHISPER_MODEL.transcribe(
            wav_buffer,
            beam_size=7,
            vad_filter=True,
            word_timestamps=False,
            initial_prompt="Trading, finance, market signals, gold, stocks, forex.",
            language="en"
        )

        transcription = " ".join([segment.text for segment in segments])
        logger.info(f"✅ Transcription successful from URL: '{transcription.strip()}'")
        return transcription.strip()

    except Exception as e:
        logger.exception(f"🚨 Transcription error from URL: {e}") # Log full traceback
        return ""


def transcribe_audio_from_bytes(audio_bytes: bytes) -> str:
    """
    Transcribes raw audio bytes (e.g., from browser's MediaRecorder, usually WebM format).
    """
    if WHISPER_MODEL is None:
        logger.error("Transcription model is not loaded for bytes transcription.")
        return "Error: Transcription model is not loaded. Cannot process audio from bytes."

    if not audio_bytes:
        logger.warning("Received empty audio bytes for transcription.")
        return "Error: Received empty audio."

    try:
        logger.info(f"Transcribing audio from raw bytes (assuming WebM format)...")
        
        audio_data_io = io.BytesIO(audio_bytes)
        
        # Added error handling for pydub's from_file and explicit format
        try:
            # Explicitly specify format as "webm" as the browser sends it
            # pydub will use ffmpeg to decode this
            audio = AudioSegment.from_file(audio_data_io, format="webm")
        except Exception as pydub_e:
            logger.error(f"🚨 Pydub failed to decode audio bytes (assumed webm). FFmpeg error: {pydub_e}")
            return "Error: Could not decode audio format. Please ensure valid WebM audio."
        
        # Check if audio is too short or silent
        if audio.duration_seconds < 0.5: # Minimum duration
            logger.warning(f"Audio too short for transcription: {audio.duration_seconds:.2f}s.")
            return "Error: Audio too short or silent."
        
        # Export as WAV in-memory to be consumed by Faster-Whisper
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        segments, info = WHISPER_MODEL.transcribe(
            wav_buffer,
            beam_size=5,
            vad_filter=True, # Voice activity detection
            word_timestamps=False,
            language="en" # Explicitly set language
        )

        transcribed_text = " ".join([segment.text for segment in segments])
        
        # Add basic check for empty transcription
        if not transcribed_text.strip():
            logger.warning("Transcription resulted in empty text.")
            return "Error: Could not transcribe clear speech."

        logger.info(f"✅ Transcription successful from bytes: '{transcribed_text.strip()}'")
        return transcribed_text.strip()

    except Exception as e:
        logger.exception(f"🚨 Error during direct audio transcription from bytes: {e}")
        return "Error processing audio for transcription."