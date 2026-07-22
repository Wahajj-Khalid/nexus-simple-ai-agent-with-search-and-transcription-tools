import os
import time
import yt_dlp
from google import genai
from groq import Groq
from tools.config import call_with_retry
from tools.knowledge_base import write_to_knowledge_base

def download_youtube_audio(video_url: str) -> tuple:
    temp_filename = f"temp_audio_{int(time.time())}"
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': f'{temp_filename}.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        ext = info.get('ext', 'm4a')
        title = info.get('title', 'Unknown Title')
        file_path = f"{temp_filename}.{ext}"
    return file_path, title

def transcribe_with_groq_whisper(file_path: str, groq_key: str) -> str:
    """Uses Groq's high-speed Whisper Large v3 model for audio transcription."""
    client = Groq(api_key=groq_key)
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, file.read()),
            model="whisper-large-v3",
            response_format="json",
        )
    return transcription.text

def transcribe_with_gemini(file_path: str, gemini_key: str) -> str:
    """Fallback transcription using Gemini File API."""
    client = genai.Client(api_key=gemini_key)
    uploaded_file = client.files.upload(file=file_path)
    
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
    client.files.delete(name=uploaded_file.name)
    return transcript_text

def transcribe_video(video_url: str) -> dict:
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    try:
        file_path, title = download_youtube_audio(video_url)
    except Exception as e:
        return {"status": "error", "message": f"Error extracting video audio: {str(e)}"}
        
    if not os.path.exists(file_path):
        return {"status": "error", "message": "Audio extraction failed. Output file not generated."}
        
    transcript_text = ""
    
    # Primary: Try Groq Whisper API first to save Gemini quota
    if groq_key and len(groq_key.strip()) > 0:
        try:
            transcript_text = transcribe_with_groq_whisper(file_path, groq_key)
        except Exception:
            transcript_text = ""
            
    # Fallback: Try Gemini if Groq was not used or failed
    if not transcript_text and gemini_key and len(gemini_key.strip()) > 0:
        try:
            transcript_text = transcribe_with_gemini(file_path, gemini_key)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return {"status": "error", "message": f"Transcription failed on both Groq and Gemini: {str(e)}"}
            
    if os.path.exists(file_path):
        os.remove(file_path)
        
    if not transcript_text:
        return {"status": "error", "message": "No valid API keys available for transcription."}
        
    write_to_knowledge_base(title, video_url, transcript_text)
    
    return {
        "status": "success",
        "title": title,
        "source_url": video_url,
        "transcript": transcript_text
    }

def transcribe_local_upload(temp_file_path: str, original_filename: str, gemini_key: str, groq_key: str = "") -> dict:
    if not os.path.exists(temp_file_path):
        return {"status": "error", "message": "Uploaded file is missing on disk."}
        
    transcript_text = ""
    
    if groq_key and len(groq_key.strip()) > 0:
        try:
            transcript_text = transcribe_with_groq_whisper(temp_file_path, groq_key)
        except Exception:
            transcript_text = ""
            
    if not transcript_text and gemini_key:
        try:
            transcript_text = transcribe_with_gemini(temp_file_path, gemini_key)
        except Exception as e:
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