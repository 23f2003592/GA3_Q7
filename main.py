import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AskRequest(BaseModel):
    video_url: str
    topic: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask")
async def find_timestamp(request: AskRequest):
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    genai.configure(api_key=gemini_key)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""Watch this YouTube video and find the timestamp when the following topic or phrase is first spoken or discussed:

Topic: {request.topic}

Return ONLY a JSON object with the timestamp in HH:MM:SS format.
Example: {{"timestamp": "00:05:47"}}
If not found, return {{"timestamp": "00:00:00"}}."""

        response = model.generate_content(
            [
                {"text": prompt},
                {"file_data": {"mime_type": "video/youtube", "file_uri": request.video_url}}
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string"}
                    },
                    "required": ["timestamp"]
                }
            )
        )

        result = json.loads(response.text)
        timestamp = result.get("timestamp", "00:00:00")

        parts = timestamp.strip().split(":")
        if len(parts) == 2:
            timestamp = f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
        elif len(parts) == 3:
            timestamp = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
        else:
            timestamp = "00:00:00"

        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
