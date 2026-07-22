import os
import re
import json
import time
from google import genai
from google.genai import types
from tools import (
    configure_apis,
    call_with_retry,
    search_knowledge_base,
    search_video,
    transcribe_video
)

def configure_environment_keys(gemini_key: str, serp_key: str, groq_key: str):
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["SERPAPI_API_KEY"] = serp_key
    os.environ["GROQ_API_KEY"] = groq_key
    configure_apis(gemini_key, serp_key)

def run_deterministic_fallback_pipeline(user_query: str):
    yield {"status": "thinking", "message": "Executing workflow pipeline..."}
    
    # Direct link check
    urls = re.findall(r'(https?://(?:www\.)?(?:youtube\.com|youtu\.be)\S+)', user_query)
    if urls:
        video_url = urls[0].strip().rstrip('.,;)')
        yield {"status": "transcribing", "message": f"Extracting audio and compiling transcript for: {video_url}"}
        result = transcribe_video(video_url)
        
        if result.get("status") == "success":
            output = f"Transcript:\n{result.get('transcript')}\n\nSource Video URL: {video_url}"
            yield {"status": "done", "message": output}
            return
        else:
            yield {"status": "done", "message": f"Error: {result.get('message')}"}
            return

    # Step 1: Search Knowledge Base
    yield {"status": "searching", "message": f"Checking local knowledge base for: '{user_query}'"}
    kb_result = search_knowledge_base(user_query)
    
    if kb_result.get("status") == "success":
        yield {"status": "searching_done", "message": "Match found in local database."}
        output = (
            f"Transcript:\n{kb_result.get('transcript')}\n\n"
            f"Source Video URL: {kb_result.get('source_url')}\n"
            f"(Retrieved from local Knowledge Base)"
        )
        yield {"status": "done", "message": output}
        return

    # Step 2: Search YouTube
    yield {"status": "searching", "message": f"Searching YouTube for: '{user_query}'"}
    found_url = search_video(user_query)
    
    if not found_url or "Error" in found_url or "No video" in found_url:
        yield {"status": "done", "message": f"Could not locate a video: {found_url}"}
        return
        
    yield {"status": "searching_done", "message": f"Found Video: {found_url}"}

    # Step 3: Transcribe Video
    yield {"status": "transcribing", "message": "Extracting audio and compiling transcript..."}
    transcribe_result = transcribe_video(found_url)
    
    if transcribe_result.get("status") == "success":
        output = f"Transcript:\n{transcribe_result.get('transcript')}\n\nSource Video URL: {found_url}"
        yield {"status": "done", "message": output}
    else:
        yield {"status": "done", "message": f"Error transcribing video: {transcribe_result.get('message')}"}

def run_agent_workflow(chat_history: list, gemini_key: str, serp_key: str, groq_key: str, model_id: str):
    configure_environment_keys(gemini_key, serp_key, groq_key)
    
    user_query = chat_history[-1]["content"] if chat_history else ""
    client = genai.Client(api_key=gemini_key)
    
    formatted_history = []
    for msg in chat_history:
        if msg["role"] == "system":
            continue
        formatted_history.append(
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
        
    strict_instruction = (
        "You are a Video Transcription Agent.\n"
        "1. Always check the local knowledge base first using search_knowledge_base.\n"
        "2. If search_knowledge_base succeeds, output the exact transcript directly without calling other tools. "
        "Append 'Source Video URL: [source_url]' and then write '(Retrieved from local Knowledge Base)' as the final line.\n"
        "3. If a YouTube URL is provided, call transcribe_video directly.\n"
        "4. If no local transcript is found and no link is provided, call search_video, then call transcribe_video.\n"
        "5. Output the transcript word-for-word without summarization.\n"
        "6. Always end responses with: 'Source Video URL: [source_url]'."
    )
    
    config = types.GenerateContentConfig(
        system_instruction=strict_instruction,
        tools=[search_knowledge_base, search_video, transcribe_video],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    )
    
    if formatted_history and formatted_history[-1].role == "user":
        formatted_history.pop()
        
    try:
        chat = client.chats.create(
            model=model_id,
            config=config,
            history=formatted_history
        )
        
        time.sleep(1.0)
        yield {"status": "thinking", "message": "Analyzing parameters..."}
        response = call_with_retry(chat.send_message, user_query)
        
        while response.function_calls:
            for call in response.function_calls:
                name = call.name
                args = call.args if call.args else {}
                
                if name == "search_knowledge_base":
                    query_term = args.get("query", "")
                    yield {"status": "searching", "message": f"Checking local database for: '{query_term}'"}
                    result_dict = search_knowledge_base(query_term)
                    result = json.dumps(result_dict)
                    yield {"status": "searching_done", "message": "Local database check complete."}
                    
                elif name == "search_video":
                    query_term = args.get("query", "")
                    yield {"status": "searching", "message": f"Searching YouTube: '{query_term}'"}
                    result = search_video(query_term)
                    yield {"status": "searching_done", "message": f"Identified Video: {result}"}
                    
                elif name == "transcribe_video":
                    video_url = args.get("video_url", "")
                    yield {"status": "transcribing", "message": "Extracting audio and compiling transcript..."}
                    result_dict = transcribe_video(video_url)
                    result = json.dumps(result_dict)
                    yield {"status": "transcribing_done", "message": "Saved transcript to Knowledge Base directory."}
                else:
                    result = f"Unknown function: {name}"
                    
                yield {"status": "thinking", "message": "Synthesizing output data..."}
                time.sleep(1.5)
                
                part = types.Part.from_function_response(
                    name=name,
                    response={'result': result}
                )
                response = call_with_retry(chat.send_message, part)
                
        yield {"status": "done", "message": response.text}
        
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str or "RESOURCE_EXHAUSTED" in err_str or "400" in err_str:
            for step in run_deterministic_fallback_pipeline(user_query):
                yield step
        else:
            raise e