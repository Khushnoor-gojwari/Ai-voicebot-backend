# # from fastapi import FastAPI, Request, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import StreamingResponse
# # from pydantic import BaseModel
# # import google.generativeai as genai
# # import requests
# # import os
# # import base64
# # import asyncio
# # import json
# # from dotenv import load_dotenv

# # # Load environment variables
# # load_dotenv()

# # app = FastAPI()

# # # Configure CORS
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["http://localhost:3000"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Configure Gemini API
# # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# # model = genai.GenerativeModel('gemini-1.5-flash')

# # # OpenAI TTS API URL
# # TTS_API_URL = "https://api.openai.com/v1/audio/speech"
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # class TranscriptionRequest(BaseModel):
# #     audio: str  # base64 encoded audio

# # class ChatRequest(BaseModel):
# #     text: str

# # @app.post("/transcribe")
# # async def transcribe_audio(request: TranscriptionRequest):
# #     try:
# #         # Decode base64 audio
# #         audio_bytes = base64.b64decode(request.audio.split(',')[1] if ',' in request.audio else request.audio)

# #         # Placeholder transcription (Replace with actual STT logic)
# #         transcription = "This is a placeholder for the transcribed text."

# #         return {"text": transcription}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # async def generate_response_stream(text):
# #     response = model.generate_content(text, stream=True)
    
# #     for chunk in response:
# #         if hasattr(chunk, 'text') and chunk.text:
# #             text_chunk = chunk.text
            
# #             # OpenAI TTS API request payload
# #             tts_payload = {
# #                 "model": "tts-1",  # Use 'tts-1-hd' for better quality
# #                 "input": text_chunk,
# #                 "voice": "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
# #             }
# #             headers = {
# #                 "Authorization": f"Bearer {OPENAI_API_KEY}",
# #                 "Content-Type": "application/json"
# #             }

# #             # Call OpenAI TTS API
# #             tts_response = requests.post(TTS_API_URL, json=tts_payload, headers=headers)
            
# #             if tts_response.status_code == 200:
# #                 audio_content = tts_response.content  # Get binary audio response
# #                 audio_bytes = base64.b64encode(audio_content).decode('utf-8')
# #                 response_dict = {"text": text_chunk, "audio": audio_bytes}
# #             else:
# #                 response_dict = {"text": text_chunk, "audio": None}

# #             # Stream JSON response
# #             yield json.dumps(response_dict) + "\n"
# #             await asyncio.sleep(0.1)

# # @app.post("/chat")
# # async def chat(request: ChatRequest):
# #     try:
# #         return StreamingResponse(
# #             generate_response_stream(request.text),
# #             media_type="application/json"
# #         )
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# import google.generativeai as genai
# import requests
# import os
# import asyncio
# import logging
# from dotenv import load_dotenv
# import time

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure Gemini API
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     logging.error("GEMINI_API_KEY is missing! Check your .env file.")
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-flash')

# # OpenAI TTS API URL
# TTS_API_URL = "https://api.openai.com/v1/audio/speech"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     logging.error("OPENAI_API_KEY is missing! Check your .env file.")

# class ChatRequest(BaseModel):
#     text: str

# async def generate_speech(text):
#     """ Generate speech from text using OpenAI TTS and return streaming response with retries. """
#     logging.info(f"Generating speech for text: {text}")

#     tts_payload = {
#         "model": "tts-1",
#         "input": text,
#         "voice": "alloy"
#     }
#     headers = {
#         "Authorization": f"Bearer {OPENAI_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             response = requests.post(TTS_API_URL, json=tts_payload, headers=headers, stream=True)
#             response.raise_for_status()  # Raise exception for HTTP errors

#             if response.status_code == 200:
#                 return StreamingResponse(response.iter_content(chunk_size=1024), media_type="audio/mpeg")
#             elif response.status_code == 429:  # Too Many Requests
#                 retry_after = int(response.headers.get("Retry-After", 5))  # Default to 5 seconds
#                 logging.warning(f"Rate limited. Retrying in {retry_after} seconds...")
#                 time.sleep(retry_after)
#             else:
#                 break  # If it's not 429, break the loop and raise an error

#         except requests.exceptions.RequestException as e:
#             logging.error(f"TTS API failed: {str(e)}")

#         # Wait before retrying
#         time.sleep(2 ** attempt)  # Exponential backoff

#     raise HTTPException(status_code=500, detail="TTS API Error: Rate limit exceeded")

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     """ Generate AI response and convert it to speech. """
#     logging.info(f"Received chat request: {request.text}")

#     try:
#         # Generate AI response
#         response = model.generate_content(request.text)
#         if not hasattr(response, "text") or not response.text:
#             logging.error("Gemini API did not return text!")
#             raise HTTPException(status_code=500, detail="Gemini API response is empty")

#         ai_text = response.text
#         logging.info(f"AI response: {ai_text}")

#         # Convert AI text response to speech
#         return await generate_speech(ai_text)

#     except Exception as e:
#         logging.error(f"Chat endpoint error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



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