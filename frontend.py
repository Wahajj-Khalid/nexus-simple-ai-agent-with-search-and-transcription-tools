import streamlit as st
import uuid
import re
import sys
import os
import time
import traceback
from backend import run_agent_workflow
from tools import transcribe_local_upload

SYSTEM_PROMPT = "You are a strict Video Transcription Agent. Find relevant videos and generate clear transcripts. Do not summarize. At the end, output 'Source Video URL: [url]'"
GREETING = "Hello. I am your Video Transcription assistant. What video or concept should we search for and transcribe today?"

MODELS = {
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.5 Flash-Lite": "gemini-3.5-flash-lite"
}

def clean_title(title: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', '', title).strip()

def clean_error_message(raw_error: str) -> str:
    err_lower = raw_error.lower()
    if "429" in err_lower or "quota" in err_lower or "resource_exhausted" in err_lower:
        return "API Rate Limit Exceeded. Switching to Groq Whisper fallback."
    if "503" in err_lower or "unavailable" in err_lower or "demand" in err_lower:
        return "The model is currently experiencing high demand. Please try again later."
    if "401" in err_lower or "403" in err_lower or "invalid" in err_lower:
        return "Invalid API configuration parameters. Please check your active keys inside Settings."
    return f"System Error: {raw_error.split('.')[0]}"

def inject_clean_styles():
    if os.path.exists("styles.css"):
        with open("styles.css", "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def run_app():
    st.set_page_config(
        page_title="Nexus", 
        layout="wide",
        page_icon="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220%22 width=%22100%22 height=%22100%22><circle cx=%2250%22 cy=%2250%22 r=%2240%22 fill=%22%2371717a%22/></svg>"
    )
    inject_clean_styles()

    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "global_model_name" not in st.session_state:
        st.session_state.global_model_name = "Gemini 3.5 Flash"
    if "generate_response" not in st.session_state:
        st.session_state.generate_response = False
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "local_file_to_process" not in st.session_state:
        st.session_state.local_file_to_process = None

    with st.sidebar:
        st.markdown("<h1 style='font-size: 2rem; margin-bottom: 0px; padding-bottom: 0px; font-weight: 700;'>Nexus</h1>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("**New Conversation**")
        
        if st.button("+ New Chat", key="new_chat_button", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {
                "title": "New Transcription Chat",
                "personality": "Video Transcription Agent",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "assistant", "content": GREETING}
                ]
            }
            st.session_state.current_chat_id = new_id
            st.session_state.generate_response = False
            st.session_state.last_error = None
            st.rerun()

        if st.session_state.chats:
            st.divider()
            st.markdown("**Recents**")
            for chat_id, chat_data in list(st.session_state.chats.items()):
                col_select, col_delete = st.columns([0.85, 0.15])
                
                is_active = (chat_id == st.session_state.current_chat_id)
                clean_name = clean_title(chat_data["title"])
                
                if col_select.button(
                    clean_name, 
                    key=f"select_{chat_id}", 
                    use_container_width=True, 
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.generate_response = False
                    st.session_state.last_error = None
                    st.rerun()
                    
                if col_delete.button("×", key=f"del_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0] if st.session_state.chats else None
                        st.session_state.generate_response = False
                        st.session_state.last_error = None
                    st.rerun()

        with st.sidebar.expander("Settings", expanded=False):
            gemini_key_secrets = st.secrets.get("GEMINI_API_KEY", "")
            serp_key_secrets = st.secrets.get("SERPAPI_API_KEY", "")
            groq_key_secrets = st.secrets.get("GROQ_API_KEY", "")
            
            if gemini_key_secrets:
                st.caption("Active Gemini key via secrets configuration.")
                gemini_api_key = gemini_key_secrets
            else:
                gemini_api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter key...")
                
            if serp_key_secrets:
                st.caption("Active SerpApi key via secrets configuration.")
                serp_api_key = serp_key_secrets
            else:
                serp_api_key = st.text_input("SerpApi API Key", type="password", placeholder="Enter key...")

            if groq_key_secrets:
                st.caption("Active Groq key via secrets configuration.")
                groq_api_key = groq_key_secrets
            else:
                groq_api_key = st.text_input("Groq API Key (Whisper Transcriber)", type="password", placeholder="Enter key...")

    if not st.session_state.current_chat_id:
        default_id = str(uuid.uuid4())
        st.session_state.chats[default_id] = {
            "title": "Welcome Chat",
            "personality": "Video Transcription Agent",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": GREETING}
            ]
        }
        st.session_state.current_chat_id = default_id

    active_chat = st.session_state.chats[st.session_state.current_chat_id]

    st.subheader(active_chat["personality"], anchor=False)
    st.caption("Finds, transcribes, and links YouTube videos using specialized tools.")
    st.divider()

    for msg in active_chat["messages"]:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Local upload transcription executor
    if st.session_state.local_file_to_process:
        uploaded_data = st.session_state.local_file_to_process
        st.session_state.local_file_to_process = None
        
        with st.chat_message("assistant"):
            status_container = st.container()
            try:
                g_key = groq_api_key if 'groq_api_key' in locals() else ""
                gem_key = gemini_api_key if 'gemini_api_key' in locals() else ""
                
                if not gem_key and not g_key:
                    st.error("Please configure an API Key inside Settings.")
                else:
                    with status_container:
                        status_widget = st.status("Caching local upload stream...", expanded=True)
                    
                    temp_path = f"temp_upload_{int(time.time())}_{uploaded_data['name']}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_data["content"])
                        
                    status_widget.update(label="File saved. Processing transcription...", state="running")
                    
                    result = transcribe_local_upload(temp_path, uploaded_data["name"], gem_key, g_key)
                    
                    if result.get("status") == "success":
                        status_widget.update(label="Complete", state="complete", expanded=False)
                        
                        final_msg = (
                            f"Successfully processed uploaded file: '{result.get('title')}'\n\n"
                            f"Transcript:\n{result.get('transcript')}\n\n"
                            f"Source Video URL: Local Upload"
                        )
                        active_chat["messages"].append({"role": "user", "content": f"Upload File: {uploaded_data['name']}"})
                        active_chat["messages"].append({"role": "assistant", "content": final_msg})
                        
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        st.rerun()
                    else:
                        status_widget.update(label="Processing Failed", state="error", expanded=True)
                        st.error(result.get("message", "Processing error occurred."))
                        
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception as e:
                st.error(clean_error_message(str(e)))

    if st.session_state.generate_response:
        with st.chat_message("assistant"):
            status_container = st.container()
            response_placeholder = st.empty()
            try:
                if not serp_api_key:
                    st.error("Please configure your SerpApi key inside the Settings panel.")
                    st.session_state.generate_response = False
                    st.stop()
                    
                active_gemini_model = MODELS[st.session_state.global_model_name]
                
                agent_stream = run_agent_workflow(
                    chat_history=active_chat["messages"],
                    gemini_key=gemini_api_key if 'gemini_api_key' in locals() else "",
                    serp_key=serp_api_key,
                    groq_key=groq_api_key if 'groq_api_key' in locals() else "",
                    model_id=active_gemini_model
                )
                
                with status_container:
                    status_widget = st.status("Initializing Agent...", expanded=True)
                
                final_text = ""
                for step in agent_stream:
                    status = step.get("status")
                    message = step.get("message")
                    
                    if status == "thinking":
                        status_widget.update(label=f"Thinking: {message}", state="running")
                    elif status == "searching":
                        status_widget.update(label=f"Searching: {message}", state="running")
                    elif status == "searching_done":
                        status_widget.write(f"Done: {message}")
                    elif status == "transcribing":
                        status_widget.update(label=f"Transcribing: {message}", state="running")
                    elif status == "transcribing_done":
                        status_widget.write(f"Done: {message}")
                    elif status == "done":
                        status_widget.update(label="Complete", state="complete", expanded=False)
                        final_text = message
                        response_placeholder.markdown(final_text)
                        
                active_chat["messages"].append({"role": "assistant", "content": final_text})
                st.session_state.last_error = None
                
            except Exception as e:
                st.session_state.last_error = clean_error_message(str(e))
                print(f"[Console API Error Exception]: {str(e)}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            finally:
                st.session_state.generate_response = False
                st.rerun()

    if st.session_state.last_error:
        st.error(st.session_state.last_error)
        if st.button("Retry last request", key="retry_button"):
            st.session_state.last_error = None
            st.session_state.generate_response = True
            st.rerun()

    with st.bottom:
        user_input = st.chat_input("How can Nexus help you today?")
        
        col_upload, col_spacer, col_gemini = st.columns([0.15, 0.58, 0.27])
        
        with col_upload:
            with st.popover("+", key="upload_popover", use_container_width=True):
                st.write("Transcribe Local Video File")
                uploaded_file = st.file_uploader("Select video file", type=["mp4", "mkv", "avi", "mov"], label_visibility="collapsed")
                if uploaded_file is not None:
                    if st.button("Transcribe", use_container_width=True, key="transcribe_local_btn"):
                        st.session_state.local_file_to_process = {
                            "name": uploaded_file.name,
                            "content": uploaded_file.getbuffer()
                        }
                        st.rerun()
            
        with col_spacer:
            st.write("")
            
        with col_gemini:
            with st.popover(st.session_state.global_model_name, key="model_popover", use_container_width=True):
                for model_name in MODELS.keys():
                    if st.button(model_name, key=f"gem_opt_{model_name}", use_container_width=True):
                        st.session_state.global_model_name = model_name
                        st.rerun()

    if user_input:
        query = user_input.strip()
        active_chat["messages"].append({"role": "user", "content": query})
        
        if len(active_chat["messages"]) <= 3:
            trimmed_title = query[:20] + ("..." if len(query) > 20 else "")
            active_chat["title"] = trimmed_title

        st.session_state.generate_response = True
        st.session_state.last_error = None
        st.rerun()

if __name__ == "__main__":
    run_app()