from fastapi import APIRouter, HTTPException
from src.app.schemas.protocol import GenerationRequest, GenerationResponse
from src.engine.generator import engine

router = APIRouter()

@router.post("/completions", response_model=GenerationResponse)
async def chat_completions(request: GenerationRequest):
    # Format Prompt for TinyLlama
    full_prompt = ""
    for msg in request.messages:
        full_prompt += f"<{msg.role}>: {msg.content}\n"
    full_prompt += "<bot>:"

    try:
        response = engine.generate(
            prompt=full_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        # Clean response
        clean_response = response.split("<bot>:")[-1].strip()
        
        return GenerationResponse(
            generated_text=clean_response,
            model=engine.model.name_or_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))