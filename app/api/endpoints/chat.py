# app/api/endpoints/chat.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import time
import logging

from app.services.rag_service import get_rag_response
from app.services.transcription_service import transcribe_audio_from_bytes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

class ChatRequest(BaseModel):
    text: str

@router.post("/direct-chat")
async def handle_direct_chat(request: ChatRequest):
    """Handles text prompts coming from the new chat UI."""
    logger.info(f"Received text chat request: '{request.text}'")
    
    rag_result = get_rag_response(request.text) 
    
    response_content = rag_result.get("response", "Sorry, I couldn't generate a response.")
    thinking_time = rag_result.get("thinking_time", "N/A") 

    return {"response": response_content, "thinking_time": thinking_time}

@router.post("/direct-audio")
async def handle_direct_audio(audio_file: UploadFile = File(...)):
    """Handles audio file uploads from the new chat UI."""
    logger.info(f"Received audio chat request for file: '{audio_file.filename}'")
    
    transcription_start_time = time.time()
    audio_bytes = await audio_file.read()
    transcribed_text = transcribe_audio_from_bytes(audio_bytes)
    transcription_end_time = time.time()
    
    logger.info(f"Transcription complete: '{transcribed_text}'")
    
    if not transcribed_text:
        raise HTTPException(status_code=400, detail="Sorry, I could not understand the audio.")

    rag_response_data = get_rag_response(transcribed_text) 

    response_content = rag_response_data.get("response", "Sorry, I couldn't generate a response.")
    rag_processing_time = rag_response_data.get("thinking_time", "N/A")

    # --- FIX: Check if rag_processing_time is a number before converting to float ---
    total_processing_time = "N/A" # Default to N/A
    try:
        # Attempt to convert to float only if it's not "N/A"
        if rag_processing_time != "N/A":
            total_processing_time = round(
                (transcription_end_time - transcription_start_time) + float(rag_processing_time),
                2
            )
        else:
            total_processing_time = "N/A" # If RAG time is N/A, total is also N/A
    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating total processing time: {e}. rag_processing_time was '{rag_processing_time}'.")
        total_processing_time = "N/A" # Ensure it remains N/A on calculation error
    # --- END FIX ---
    
    return {
        "transcribed_text": transcribed_text,
        "response": response_content,
        "thinking_time": total_processing_time
    }