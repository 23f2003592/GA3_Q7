import os, sys, json, time, traceback, tempfile, shutil
from io import StringIO
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import google.generativeai as genai

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AskRequest(BaseModel):
    video_url: str
    topic: str

class CommentRequest(BaseModel):
    comment: str

class CodeRequest(BaseModel):
    code: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask")
async def find_timestamp(request: AskRequest):
    import yt_dlp
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    genai.configure(api_key=gemini_key)
    tmp_dir = tempfile.mkdtemp()
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([request.video_url])

        audio_file = None
        for f in os.listdir(tmp_dir):
            if f.endswith(".mp3") or f.endswith(".m4a") or f.endswith(".webm"):
                audio_file = os.path.join(tmp_dir, f)
                break
        if not audio_file:
            raise HTTPException(status_code=500, detail="Audio download failed")

        uploaded = genai.upload_file(audio_file, mime_type="audio/mpeg")
        max_wait = 120
        waited = 0
        while uploaded.state.name != "ACTIVE" and waited < max_wait:
            time.sleep(3)
            waited += 3
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise HTTPException(status_code=500, detail="File upload did not become active")

        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""Listen to this audio carefully and find when the following topic or phrase is first spoken or discussed:

Topic: {request.topic}

Return ONLY a JSON with the timestamp in HH:MM:SS format when this topic first appears.
If not found, return "00:00:00"."""

        response = model.generate_content(
            [uploaded, prompt],
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

        try:
            genai.delete_file(uploaded.name)
        except:
            pass

        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

