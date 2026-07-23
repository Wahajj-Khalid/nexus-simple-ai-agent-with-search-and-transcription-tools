import os
import re
import time
import requests
import yt_dlp
from google import genai
from google.genai import types
from groq import Groq
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from tools.config import call_with_retry
from tools.knowledge_base import write_to_knowledge_base


def extract_video_id(url: str) -> str:
    """Extracts YouTube video ID from various YouTube URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/|v\/|youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def transcribe_via_gemini_uri(video_url: str, gemini_key: str) -> str:
    """Passes YouTube video URL directly to Google Gemini for native cloud processing."""
    client = genai.Client(api_key=gemini_key)
    prompt = (
        "Provide a complete, chronologically structured transcription of this video. "
        "Output only the transcription text. Do not summarize or edit."
    )
    response = call_with_retry(
        client.models.generate_content,
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_uri(
                file_uri=video_url,
                mime_type="video/mp4"
            ),
            prompt
        ]
    )
    return response.text


def download_youtube_audio(video_url: str) -> tuple:
    """Downloads YouTube audio locally using yt-dlp."""
    temp_filename = f"temp_audio_{int(time.time())}"
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': f'{temp_filename}.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'mweb'],
                'skip': ['hls', 'dash']
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

    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass

    return transcript_text


def fetch_youtube_english_captions(video_id: str) -> str:
    """Fetches English manual or auto-generated captions directly from YouTube."""
    ytt_api = YouTubeTranscriptApi()

    if hasattr(ytt_api, "list"):
        transcript_list = ytt_api.list(video_id)
    elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    else:
        fetched_data = ytt_api.fetch(video_id, languages=['en', 'en-US'])
        return " ".join([item['text'] for item in fetched_data])

    try:
        transcript = transcript_list.find_manually_created_transcript(['en', 'en-US'])
        return " ".join([item['text'] for item in transcript.fetch()])
    except NoTranscriptFound:
        pass

    try:
        transcript = transcript_list.find_generated_transcript(['en', 'en-US'])
        return " ".join([item['text'] for item in transcript.fetch()])
    except NoTranscriptFound:
        pass

    raise NoTranscriptFound(video_id, ['en', 'en-US'], transcript_list)


def fetch_transcript_via_serpapi(video_id: str, serp_key: str) -> str:
    """Queries SerpApi's dedicated youtube_video_transcript engine using the Video ID."""
    if not serp_key or not video_id:
        return ""
        
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "youtube_video_transcript",
        "v": video_id,
        "api_key": serp_key
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            transcript_list = data.get("transcript", [])
            if transcript_list:
                parts = [item.get("snippet", "") or item.get("text", "") for item in transcript_list]
                full_text = " ".join([p for p in parts if p])
                if full_text and len(full_text.strip()) > 0:
                    return full_text
    except Exception:
        pass
    return ""


def transcribe_video(video_url: str) -> dict:
    """
    Agent Tool Function:
    1. Tries direct Gemini Native URI Ingestion FIRST.
    2. Falls back to yt-dlp audio download + Gemini/Groq model transcription.
    3. Falls back to English YouTube captions via youtube-transcript-api.
    4. Falls back to SerpApi youtube_video_transcript engine.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    serp_key = os.getenv("SERPAPI_API_KEY", "")

    transcript_text = ""
    title = f"YouTube Video ({video_url})"

    # Step 1: Direct Gemini Native URI Ingestion FIRST
    if gemini_key and len(gemini_key.strip()) > 0:
        try:
            transcript_text = transcribe_via_gemini_uri(video_url, gemini_key)
        except Exception:
            transcript_text = ""

    if transcript_text and transcript_text.strip():
        write_to_knowledge_base(title, video_url, transcript_text)
        return {
            "status": "success",
            "title": title,
            "source_url": video_url,
            "transcript": transcript_text
        }

    # Step 2: Attempt direct audio extraction via yt-dlp -> AI Model
    file_path = None
    try:
        file_path, downloaded_title = download_youtube_audio(video_url)
        if downloaded_title and downloaded_title != "Unknown Title":
            title = downloaded_title

        if gemini_key and len(gemini_key.strip()) > 0:
            try:
                transcript_text = transcribe_with_gemini_file(file_path, gemini_key)
            except Exception:
                transcript_text = ""

        if not transcript_text and groq_key and len(groq_key.strip()) > 0:
            try:
                transcript_text = transcribe_with_groq_whisper(file_path, groq_key)
            except Exception:
                transcript_text = ""

    except Exception:
        pass
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

    if transcript_text and transcript_text.strip():
        write_to_knowledge_base(title, video_url, transcript_text)
        return {
            "status": "success",
            "title": title,
            "source_url": video_url,
            "transcript": transcript_text
        }

    # Step 3: Fallback to YouTube captions via youtube-transcript-api
    video_id = extract_video_id(video_url)
    if not video_id:
        return {
            "status": "error",
            "message": f"Invalid YouTube URL format: '{video_url}'"
        }

    try:
        caption_text = fetch_youtube_english_captions(video_id)
        if caption_text and caption_text.strip():
            write_to_knowledge_base(title, video_url, caption_text)
            return {
                "status": "success",
                "title": title,
                "source_url": video_url,
                "transcript": caption_text
            }
    except Exception:
        pass

    # Step 4: Fallback to SerpApi youtube_video_transcript engine
    if serp_key:
        serp_text = fetch_transcript_via_serpapi(video_id, serp_key)
        if serp_text and serp_text.strip():
            write_to_knowledge_base(title, video_url, serp_text)
            return {
                "status": "success",
                "title": title,
                "source_url": video_url,
                "transcript": serp_text
            }

    # Step 5: Final error state if all 4 extraction pathways failed
    return {
        "status": "error",
        "message": "Unable to retrieve video transcript via Gemini URI, Audio Download, YouTube Captions, or SerpApi."
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