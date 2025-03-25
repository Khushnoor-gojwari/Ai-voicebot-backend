



from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
import asyncio
import logging
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY is missing! Check your .env file.")
    
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

class ChatRequest(BaseModel):
    text: str
    source_language: str = "en-US"
    target_language: str = "en-US"

async def generate_response_stream(text, source_lang="en-US", target_lang="en-US"):
    """Generate streaming response with larger chunks to avoid cutting off speech."""
    context_prompt = f"You are an AI assistant responding in the context of a {target_lang} language conversation. "
    full_prompt = context_prompt + text
    
    try:
        response = model.generate_content(full_prompt, stream=True)
        
        buffer = []  # Accumulate larger chunks before sending
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                buffer.append(chunk.text)
                
                # Send when we reach a full sentence or a long enough chunk
                if len(" ".join(buffer)) > 150 or chunk.text.endswith(('.', '?', '!')):
                    text_chunk = " ".join(buffer)
                    buffer = []  # Reset buffer
                    
                    yield json.dumps({
                        "text": text_chunk, 
                        "source_language": source_lang,
                        "target_language": target_lang
                    }) + "\n"

                    await asyncio.sleep(0.3)  # Ensure smooth streaming
    
    except Exception as e:
        logging.error(f"Response generation error: {e}")
        yield json.dumps({
            "text": f"Sorry, there was an error processing your request: {str(e)}",
            "source_language": source_lang,
            "target_language": target_lang
        }) + "\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with language support"""
    try:
        return StreamingResponse(
            generate_response_stream(
                request.text, 
                source_lang=request.source_language,  # ✅ Fixed parameter name
                target_lang=request.target_language   # ✅ Fixed parameter name
            ),
            media_type="application/json"
        )
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
