from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_service import get_rag_response

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/prompt")
def handle_prompt_request(request: PromptRequest):
    """
    Receives a prompt from the mini-frontend and gets a response from the RAG service.
    """
    response_text = get_rag_response(request.prompt)
    return {"response": response_text}