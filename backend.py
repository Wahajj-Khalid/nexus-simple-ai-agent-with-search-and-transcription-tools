import os
import re
import time
import requests
import yt_dlp
import google.generativeai as genai

def configure_apis(gemini_key: str, serp_key: str):
    """Configures environment and SDK keys."""
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["SERPAPI_API_KEY"] = serp_key
    genai.configure(api_key=gemini_key)

def search_video(query: str) -> str:
    """
    Searches for a video on YouTube using SerpApi.
    Returns the first matching video URL.
    """
    serp_key = os.getenv("SERPAPI_API_KEY", "")
    if not serp_key:
        return "Error: SerpApi API Key is not configured."
        
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": serp_key
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        video_results = data.get("video_results", [])
        if video_results:
            return video_results[0].get("link", "No link found in results.")
        return "No video results found."
    except Exception as e:
        return f"Error searching video: {str(e)}"

def transcribe_video(video_url: str) -> str:
    """
    Downloads raw YouTube audio, uploads it to Gemini for transcription,
    and saves the complete transcript to a file in the Knowledge Base.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        return "Error: Gemini API Key is not configured."
        
    temp_filename = f"temp_audio_{int(time.time())}"
    
    # Best-audio m4a download prevents the strict need for external ffmpeg builds
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': f'{temp_filename}.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            ext = info.get('ext', 'm4a')
            title = info.get('title', 'Unknown Title')
            file_path = f"{temp_filename}.{ext}"
    except Exception as e:
        return f"Error extracting video audio: {str(e)}"
        
    if not os.path.exists(file_path):
        return "Error: Audio extraction failed. Output file not generated."
        
    try:
        # Upload the audio file to Gemini's File API
        uploaded_file = genai.upload_file(path=file_path)
        
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            raise Exception("File processing failed on Gemini's API servers.")
            
        # Call Gemini for transcription
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Provide a complete, chronologically structured transcription of this audio. "
            "Identify speakers where distinct, and maintain high text fidelity."
        )
        response = model.generate_content([uploaded_file, prompt])
        transcript_text = response.text
        
        # Clean up file on Google Cloud storage
        genai.delete_file(uploaded_file.name)
        
    except Exception as e:
        return f"Error transcribing audio content: {str(e)}"
    finally:
        # Clean up local temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    # Save output to Knowledge Base
    try:
        os.makedirs("knowledge_base", exist_ok=True)
        safe_title = re.sub(r'[^\w\-_\. ]', '_', title)
        kb_path = os.path.join("knowledge_base", f"{safe_title}.txt")
        with open(kb_path, "w", encoding="utf-8") as f:
            f.write(f"Source URL: {video_url}\n")
            f.write(f"Title: {title}\n")
            f.write("=" * 50 + "\n\n")
            f.write(transcript_text)
            
        return (
            f"Successfully processed video: '{title}'\n"
            f"Saved to Knowledge Base path: '{kb_path}'\n\n"
            f"Transcript Preview:\n{transcript_text[:1200]}..."
        )
    except Exception as e:
        return f"Error writing file to knowledge base: {str(e)}"

def run_agent_workflow(chat_history: list, gemini_key: str, serp_key: str, model_id: str):
    """
    Main agent execution generator. Intercepts tool execution to provide
    real-time structured step indicators back to the Streamlit UI.
    """
    configure_apis(gemini_key, serp_key)
    
    # Process history into Gemini-accepted formats (system prompt separate)
    system_instruction = "You are an advanced AI Video Assistant."
    formatted_history = []
    
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_instruction = content
            continue
        formatted_history.append({
            "role": "user" if role == "user" else "model",
            "parts": [content]
        })
        
    model = genai.GenerativeModel(
        model_name=model_id,
        tools=[search_video, transcribe_video],
        system_instruction=system_instruction
    )
    
    # Separate current query for dynamic tracking
    user_query = ""
    if formatted_history and formatted_history[-1]["role"] == "user":
        user_query = formatted_history.pop()["parts"][0]
        
    chat = model.start_chat(history=formatted_history)
    
    yield {"status": "thinking", "message": "Analyzing request parameters..."}
    response = chat.send_message(user_query)
    
    # Loop over tools as long as Gemini requests them
    while response.function_calls:
        for call in response.function_calls:
            name = call.name
            args = dict(call.args)
            
            if name == "search_video":
                query_term = args.get("query", "")
                yield {"status": "searching", "message": f"Searching YouTube: '{query_term}'"}
                result = search_video(query_term)
                yield {"status": "searching_done", "message": f"Identified Video: {result}"}
                
            elif name == "transcribe_video":
                video_url = args.get("video_url", "")
                yield {"status": "transcribing", "message": "Extracting audio and compiling transcript..."}
                result = transcribe_video(video_url)
                yield {"status": "transcribing_done", "message": "Saved transcript to Knowledge Base directory."}
            else:
                result = f"Unknown system function: {name}"
                
            yield {"status": "thinking", "message": "Synthesizing output data..."}
            
            # Feed function results back into chat loop
            response = chat.send_message(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name,
                        response={'result': result}
                    )
                )
            )
            
    yield {"status": "done", "message": response.text}