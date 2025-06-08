from fastapi import APIRouter, Form
from twilio.rest import Client
from app.core.config import settings
from app.services.rag_service import get_rag_response
from app.services.transcription_service import transcribe_audio_from_url

router = APIRouter()

# This is the main, authenticated Twilio client for our application
twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

@router.post("/whatsapp")
async def handle_whatsapp_message(
    From: str = Form(...),
    Body: str = Form(None),
    NumMedia: int = Form(0),
    MediaUrl0: str = Form(None)
):
    processed_text = ""
    response_message = "Please send a text or voice message."
    
    if NumMedia > 0 and MediaUrl0:
        print("Received an audio message.")
        
        # --- THIS IS THE FIX ---
        # We now pass the authenticated http_client into the transcription service
        processed_text = transcribe_audio_from_url(MediaUrl0)
        
        if not processed_text:
            response_message = "Sorry, I couldn't understand the audio. Please try again."
    elif Body:
        print("Received a text message.")
        processed_text = Body

    if processed_text:
        print(f"Processing text: '{processed_text}'")
        response_message = get_rag_response(processed_text)
    
    print(f"Sending reply to {From}: {response_message}")
    twilio_client.messages.create(
        from_=settings.TWILIO_PHONE_NUMBER,
        body=response_message,
        to=From
    )
    
    return {"status": "message processed"}