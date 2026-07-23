import os
import time
import yt_dlp
from google import genai
from google.genai import types
from groq import Groq
from tools.config import call_with_retry
from tools.knowledge_base import write_to_knowledge_base

def download_youtube_audio(video_url: str) -> tuple:
    """
    Downloads YouTube audio locally using yt-dlp.
    Includes custom extractor args to bypass Streamlit Cloud (403 Forbidden) blocks.
    """
    temp_filename = f"temp_audio_{int(time.time())}"
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': f'{temp_filename}.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        # Headers & client overrides required to prevent 403 Forbidden on cloud servers (Streamlit/AWS)
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'mweb', 'web_embedded'],
            }
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        ext = info.get('ext', 'm4a')
        title = info.get('title', 'Unknown Title')
        file_path = f"{temp_filename}.{ext}"
    return file_path, title

def transcribe_with_groq_whisper(file_path: str, groq_key: str) -> str:
    """Transcribes audio file using Groq Whisper API."""
    client = Groq(api_key=groq_key)
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, file.read()),
            model="whisper-large-v3",
            response_format="json",
        )
    return transcription.text

def transcribe_with_gemini_file(file_path: str, gemini_key: str) -> str:
    """Uploads audio file to Gemini File API and transcribes it using Gemini 3.5 Flash."""
    client = genai.Client(api_key=gemini_key)
    uploaded_file = client.files.upload(file=file_path)
    
    # Wait for Gemini audio processing to complete
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
        
    if uploaded_file.state.name == "FAILED":
        raise Exception("File processing failed on Gemini servers.")
        
    prompt = (
        "Provide a complete, chronologically structured transcription of this audio. "
        "Output only the transcription text. Do not summarize or edit."
    )
    
    response = call_with_retry(
        client.models.generate_content,
        model="gemini-3.5-flash",
        contents=[uploaded_file, prompt]
    )
    
    transcript_text = response.text
    
    # Clean up uploaded file from Gemini storage
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass
        
    return transcript_text

def transcribe_video(video_url: str) -> dict:
    """
    Agent Tool Function: Downloads YouTube audio via yt-dlp and transcribes it 
    using Gemini API (or Groq Whisper fallback).
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    transcript_text = ""
    title = f"YouTube Video ({video_url})"
    file_path = None
    
    # Step 1: Download audio using cloud-compatible yt-dlp config
    try:
        file_path, downloaded_title = download_youtube_audio(video_url)
        if downloaded_title and downloaded_title != "Unknown Title":
            title = downloaded_title
    except Exception as e:
        return {
            "status": "error",
            "message": f"HTTP Error 403: Forbidden when attempting to access video. Details: {str(e)}"
        }

    # Step 2: Transcribe downloaded audio using Gemini API
    if gemini_key and len(gemini_key.strip()) > 0:
        try:
            transcript_text = transcribe_with_gemini_file(file_path, gemini_key)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                return {
                    "status": "error",
                    "message": "Gemini API Rate Limit Exceeded. Quota limits reached."
                }
            transcript_text = ""

    # Step 3: Fallback to Groq Whisper if Gemini fails or key is missing
    if not transcript_text and groq_key and len(groq_key.strip()) > 0:
        try:
            transcript_text = transcribe_with_groq_whisper(file_path, groq_key)
        except Exception:
            transcript_text = ""

    # Cleanup local temporary file
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    if not transcript_text:
        return {
            "status": "error",
            "message": "Failed to transcribe audio with the available API keys."
        }
        
    write_to_knowledge_base(title, video_url, transcript_text)
    
    return {
        "status": "success",
        "title": title,
        "source_url": video_url,
        "transcript": transcript_text
    }

def transcribe_local_upload(temp_file_path: str, original_filename: str, gemini_key: str, groq_key: str = "") -> dict:
    """Agent Tool Function: Transcribes manually uploaded local audio files."""
    if not os.path.exists(temp_file_path):
        return {"status": "error", "message": "Uploaded file is missing on disk."}
        
    transcript_text = ""
    
    if gemini_key:
        try:
            transcript_text = transcribe_with_gemini_file(temp_file_path, gemini_key)
        except Exception as e:
            if groq_key:
                try:
                    transcript_text = transcribe_with_groq_whisper(temp_file_path, groq_key)
                except Exception:
                    pass
            if not transcript_text:
                return {"status": "error", "message": f"Error transcribing file: {str(e)}"}
            
    if not transcript_text:
        return {"status": "error", "message": "Transcription service unavailable."}
        
    write_to_knowledge_base(original_filename, "Local Upload", transcript_text)
    
    return {
        "status": "success",
        "title": original_filename,
        "source_url": "Local Upload",
        "transcript": transcript_text
    }