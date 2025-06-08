from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from app.services.rag_service import get_rag_response
from app.services.transcription_service import transcribe_audio_from_bytes

router = APIRouter()

class ChatRequest(BaseModel):
    text: str

@router.post("/direct-chat")
async def handle_direct_chat(request: ChatRequest):
    """Handles text prompts coming from the new chat UI."""
    response_text = get_rag_response(request.text)
    return {"response": response_text}

@router.post("/direct-audio")
async def handle_direct_audio(audio_file: UploadFile = File(...)):
    """Handles audio file uploads from the new chat UI."""
    audio_bytes = await audio_file.read()
    
    # Transcribe the audio bytes first
    transcribed_text = transcribe_audio_from_bytes(audio_bytes)
    
    if not transcribed_text:
        return {"response": "Sorry, I could not understand the audio."}
        
    # Get the final response from the RAG service
    response_text = get_rag_response(transcribed_text)
    return {"response": response_text}