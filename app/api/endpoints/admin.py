from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path

router = APIRouter()
PROMPT_FILE_PATH = Path("app/prompt_template.txt")

class PromptUpdateRequest(BaseModel):
    new_prompt: str

@router.get("/admin/prompt", summary="Get the current RAG prompt template")
def get_prompt():
    """Reads and returns the content of the master prompt template file."""
    if not PROMPT_FILE_PATH.is_file():
        raise HTTPException(status_code=404, detail="Prompt template file not found.")
    return {"prompt": PROMPT_FILE_PATH.read_text()}

@router.post("/admin/prompt", summary="Update the RAG prompt template")
def update_prompt(request: PromptUpdateRequest):
    """Overwrites the master prompt template file with new content."""
    try:
        PROMPT_FILE_PATH.write_text(request.new_prompt)
        return {"status": "success", "message": "Prompt template updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write to prompt file: {e}")