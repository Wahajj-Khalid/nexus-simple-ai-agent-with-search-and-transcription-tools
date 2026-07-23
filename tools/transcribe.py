import os
import re
import time
import yt_dlp
from google import genai
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
    """
    Fetches English (en, en-US) manual or auto-generated captions directly from YouTube API.
    Strictly ignores all other languages without fallback.
    """
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # 1. Try manual English transcripts first
    try:
        transcript = transcript_list.find_manually_created_transcript(['en', 'en-US'])
        return " ".join([item['text'] for item in transcript.fetch()])
    except NoTranscriptFound:
        pass

    # 2. Try auto-generated English transcripts
    try:
        transcript = transcript_list.find_generated_transcript(['en', 'en-US'])
        return " ".join([item['text'] for item in transcript.fetch()])
    except NoTranscriptFound:
        pass

    raise NoTranscriptFound(video_id, ['en', 'en-US'], transcript_list)


def transcribe_video(video_url: str) -> dict:
    """
    1. First attempts audio download + Gemini/Groq model transcription.
    2. If audio download fails (e.g. HTTP 403), falls back to YouTube English captions.
    3. If all fail, returns a clean error message.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

    transcript_text = ""
    title = f"YouTube Video ({video_url})"
    file_path = None

    # --- PRIMARY METHOD: Direct Audio Download & Model Transcription ---
    try:
        file_path, downloaded_title = download_youtube_audio(video_url)
        if downloaded_title and downloaded_title != "Unknown Title":
            title = downloaded_title

        # Step 2a: Try Gemini
        if gemini_key and len(gemini_key.strip()) > 0:
            try:
                transcript_text = transcribe_with_gemini_file(file_path, gemini_key)
            except Exception:
                transcript_text = ""

        # Step 2b: Try Groq Whisper Fallback
        if not transcript_text and groq_key and len(groq_key.strip()) > 0:
            try:
                transcript_text = transcribe_with_groq_whisper(file_path, groq_key)
            except Exception:
                transcript_text = ""

    except Exception:
        # Audio download failed (HTTP 403 Forbidden on cloud platform)
        pass
    finally:
        # Always clean up temporary audio file if created
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

    # If audio transcription succeeded, return immediately
    if transcript_text and transcript_text.strip():
        write_to_knowledge_base(title, video_url, transcript_text)
        return {
            "status": "success",
            "title": title,
            "source_url": video_url,
            "transcript": transcript_text
        }

    # --- SECONDARY METHOD: YouTube English Captions Fallback ---
    video_id = extract_video_id(video_url)
    if not video_id:
        return {
            "status": "error",
            "message": "Invalid YouTube URL format provided."
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
    except TranscriptsDisabled:
        return {
            "status": "error",
            "message": "Transcripts are disabled for this video on YouTube."
        }
    except NoTranscriptFound:
        return {
            "status": "error",
            "message": "No English transcript (manual or auto-generated) is available for this video."
        }
    except VideoUnavailable:
        return {
            "status": "error",
            "message": "The YouTube video is unavailable or private."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Transcription failed: {str(e)}"
        }

    return {
        "status": "error",
        "message": "Failed to transcribe video using available services."
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