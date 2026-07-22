import streamlit as st
import uuid
import re
import sys
import traceback
from backend import run_agent_workflow

# Adapt system profiles to video-focused specializations
PERSONALITIES = {
    "Auto-Transcriptionist": {
        "description": "Searches for a video and outputs a full structured transcript.",
        "system_prompt": "You are an automated video transcription specialist. Your goal is to find relevant videos and generate clear, structured, and complete transcriptions.",
        "greeting": "Hello! Provide a topic or a YouTube query, and I'll find the video and transcribe it for you."
    },
    "Video Summarizer": {
        "description": "Transcribes and provides an elegant executive summary of the content.",
        "system_prompt": "You are a professional Video Summarizer. First transcribe the video, then provide a structured executive summary highlighting core takeaways.",
        "greeting": "Hello! Give me a search term or a direct link, and I will find, transcribe, and synthesize an elegant summary of the video."
    },
    "Study Guide Creator": {
        "description": "Generates key questions, definitions, and cheat sheets from the video.",
        "system_prompt": "You are an educational curriculum creator. Use the tools to find and transcribe relevant educational videos, then compile high-quality study guides from them.",
        "greeting": "Hello! What educational concept or topic are we turning into a structured study guide today?"
    }
}

MODELS = {
    "Gemini 1.5 Flash (Fast)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Precise)": "gemini-1.5-pro",
    "Gemini 2.5 Flash (Balanced)": "gemini-2.5-flash"
}

def clean_title(title: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', '', title).strip()

def inject_clean_styles():
    st.markdown("""
        <style>
        /* INTERCEPT STREAMLIT CORE ACCENTS */
        :root, [data-testid="stAppViewContainer"] {
            --primary-color: #71717a !important;
        }
        
        .block-container {
            padding-top: 1.5rem !important;
            max-width: 800px !important;
        }
        
        .element-container h3 a, .element-container h2 a {
            display: none !important;
        }
        
        hr, div[data-testid="stDivider"] {
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
        
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-bottom: 90px !important; 
            position: relative !important;
            height: calc(100vh - 2rem) !important;
        }

        /* DYNAMIC SETTINGS EXPANDER WITH DEFINED BORDERS */
        div[data-testid="stSidebar"] div.stExpander {
            position: absolute !important;
            bottom: 1rem !important;
            left: 1rem !important;
            right: 1rem !important;
            background-color: var(--background-color) !important;
            border: 1px solid rgba(113, 113, 122, 0.4) !important;
            border-radius: 6px !important;
            z-index: 999;
        }
        
        div[data-testid="stSidebar"] div.stExpander details summary {
            font-size: 0.9rem !important;
        }
        
        /* INACTIVE SIDEBAR ITEMS */
        section[data-testid="stSidebar"] button[kind="primary"],
        section[data-testid="stSidebar"] button[kind="secondary"] {
            background-color: transparent !important;
            color: var(--text-color) !important;
            opacity: 0.75;
            border: 1px solid rgba(113, 113, 122, 0.2) !important;
            box-shadow: none !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            width: 100% !important;
            padding: 0.5rem 0.75rem !important;
            border-radius: 6px !important;
            margin-bottom: 2px !important;
        }
        
        section[data-testid="stSidebar"] button[kind="primary"] div,
        section[data-testid="stSidebar"] button[kind="primary"] p,
        section[data-testid="stSidebar"] button[kind="primary"] span,
        section[data-testid="stSidebar"] button[kind="secondary"] div,
        section[data-testid="stSidebar"] button[kind="secondary"] p,
        section[data-testid="stSidebar"] button[kind="secondary"] span {
            text-align: left !important;
            justify-content: flex-start !important;
            align-items: center !important;
            display: inline-flex !important;
            width: auto !important;
            margin: 0 !important;
        }
        
        /* ACTIVE SIDEBAR ITEM */
        section[data-testid="stSidebar"] div[class*="st-key-select_"] button[kind="primary"] {
            background-color: rgba(113, 113, 122, 0.15) !important;
            color: var(--text-color) !important;
            opacity: 1 !important;
            border: 1px solid rgba(113, 113, 122, 0.5) !important;
            border-left: 4px solid #71717a !important;
        }
        
        section[data-testid="stSidebar"] button[kind="primary"]:hover,
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: rgba(113, 113, 122, 0.1) !important;
            border-color: rgba(113, 113, 122, 0.4) !important;
            opacity: 1 !important;
        }
        
        section[data-testid="stSidebar"] div[class*="st-key-del_"] button {
            color: var(--text-color) !important;
            opacity: 0.4 !important;
            font-size: 1.2rem !important;
            text-align: center !important;
            justify-content: center !important;
            padding: 0px !important;
            border: none !important;
        }
        
        section[data-testid="stSidebar"] div[class*="st-key-del_"] button:hover {
            color: #ef4444 !important;
            opacity: 1 !important;
            background-color: transparent !important;
            border: none !important;
        }

        /* PILL POP-OVERS WITH ENHANCED BORDERS */
        div.st-key-model_popover button,
        section[data-testid="stSidebar"] div[class*="st-key-persona_popover"] button {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
            border: 1px solid rgba(113, 113, 122, 0.45) !important;
            border-radius: 20px !important;
            padding: 0.3rem 0.8rem !important;
            font-size: 0.85rem !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-weight: 500 !important;
            width: 100% !important;
            box-shadow: none !important;
            height: 38px !important;
        }
        
        div.st-key-model_popover button:hover,
        section[data-testid="stSidebar"] div[class*="st-key-persona_popover"] button:hover {
            background-color: rgba(113, 113, 122, 0.08) !important;
            border-color: rgba(113, 113, 122, 0.7) !important;
        }
        
        /* "+ NEW CHAT" WIDGET */
        section[data-testid="stSidebar"] div[class*="st-key-new_chat_button"] button {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
            border: 1px solid rgba(113, 113, 122, 0.45) !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            height: 40px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            box-shadow: none !important;
        }
        
        section[data-testid="stSidebar"] div[class*="st-key-new_chat_button"] button:hover {
            background-color: rgba(113, 113, 122, 0.08) !important;
            border-color: rgba(113, 113, 122, 0.7) !important;
        }

        /* REMOVE CHAT AVATARS */
        div[data-testid="stChatMessageAvatar"],
        div[data-testid="stChatMessageAvatarUser"],
        div[data-testid="stChatMessageAvatarAssistant"],
        .stChatMessage [data-testid="chatAvatarIcon-user"],
        .stChatMessage [data-testid="chatAvatarIcon-assistant"],
        .stChatMessage img {
            display: none !important;
            width: 0px !important; height: 0px !important;
            margin: 0px !important; padding: 0px !important;
        }
        
        div[data-testid="stChatMessageContent"] {
            padding-left: 0px !important;
            margin-left: 0px !important;
            width: 100% !important;
        }

        /* HIGHER VISIBILITY USER QUESTION BLOCK */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background-color: rgba(113, 113, 122, 0.15) !important;
            border: 1px solid rgba(113, 113, 122, 0.3) !important;
            border-radius: 8px !important;
            padding: 0.85rem 1.2rem !important;
            margin-bottom: 1.2rem !important;
            margin-left: auto !important;
            max-width: 80% !important;
        }
        
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {
            text-align: right !important;
            color: var(--text-color) !important;
        }
        
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
            background-color: transparent !important;
            padding: 1rem 0rem !important;
            margin-bottom: 1rem !important;
        }

        /* CHAT INPUT FIELD DEFINED BORDERS */
        [data-testid="stChatInput"] {
            border: 1px solid rgba(113, 113, 122, 0.5) !important;
            background-color: var(--background-color) !important;
            border-radius: 8px !important;
        }
        
        [data-testid="stChatInput"]:focus-within {
            border-color: #71717a !important;
            box-shadow: 0 0 0 1px #71717a !important;
        }
        
        [data-testid="stChatInput"] textarea:focus {
            border-color: transparent !important;
            box-shadow: none !important;
            outline: none !important;
        }
        
        [data-testid="stChatInput"] textarea {
            box-shadow: none !important;
            color: var(--text-color) !important;
        }
        
        /* STABLE LIGHT/DARK DISMISSAL AND SUBMIT ACTIONS */
        [data-testid="stChatInput"] button:disabled {
            background-color: transparent !important;
            color: var(--text-color) !important;
            opacity: 0.35 !important;
        }
        
        [data-testid="stChatInput"] button:not(:disabled) {
            background-color: #71717a !important;
            color: #ffffff !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button:not(:disabled):hover {
            background-color: #52525b !important;
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

def run_app():
    st.set_page_config(
        page_title="Nexus Video AI", 
        layout="wide",
        page_icon="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220%22 width=%22100%22 height=%22100%22><circle cx=%2250%22 cy=%2250%22 r=%2240%22 fill=%22%2371717a%22/></svg>"
    )
    inject_clean_styles()

    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "global_model_name" not in st.session_state:
        st.session_state.global_model_name = "Gemini 1.5 Flash (Fast)"
    if "selected_persona_setup" not in st.session_state:
        st.session_state.selected_persona_setup = "Auto-Transcriptionist"
    if "generate_response" not in st.session_state:
        st.session_state.generate_response = False

    # -------------------------------------------------------------------------
    # Sidebar Setup
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("<h1 style='font-size: 2rem; margin-bottom: 0px; padding-bottom: 0px; font-weight: 700;'>Nexus</h1>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("**New Agent Mode**")
        
        with st.popover(st.session_state.selected_persona_setup, key="persona_popover", use_container_width=True):
            for persona in PERSONALITIES.keys():
                if st.button(persona, key=f"p_opt_{persona}", use_container_width=True):
                    st.session_state.selected_persona_setup = persona
                    st.rerun()
        
        if st.button("+ New Chat", key="new_chat_button", use_container_width=True):
            new_id = str(uuid.uuid4())
            persona_data = PERSONALITIES[st.session_state.selected_persona_setup]
            
            st.session_state.chats[new_id] = {
                "title": f"New {st.session_state.selected_persona_setup} Chat",
                "personality": st.session_state.selected_persona_setup,
                "messages": [
                    {"role": "system", "content": persona_data["system_prompt"]},
                    {"role": "assistant", "content": persona_data["greeting"]}
                ]
            }
            st.session_state.current_chat_id = new_id
            st.session_state.generate_response = False
            st.rerun()

        # History list
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
                    st.rerun()
                    
                if col_delete.button("×", key=f"del_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0] if st.session_state.chats else None
                        st.session_state.generate_response = False
                    st.rerun()

        # Input field configurations inside Settings expander
        with st.sidebar.expander("Settings", expanded=False):
            gemini_key_secrets = st.secrets.get("GEMINI_API_KEY", "")
            serp_key_secrets = st.secrets.get("SERPAPI_API_KEY", "")
            
            if gemini_key_secrets:
                st.caption("Gemini API Key: configured via secrets.")
                gemini_api_key = gemini_key_secrets
            else:
                gemini_api_key = st.text_input("Gemini API Key", type="password", placeholder="AI Studio Key...")
                
            if serp_key_secrets:
                st.caption("SerpApi Key: configured via secrets.")
                serp_api_key = serp_key_secrets
            else:
                serp_api_key = st.text_input("SerpApi Key", type="password", placeholder="SerpApi Key...")

    # Fallback default chat creator
    if not st.session_state.current_chat_id:
        default_id = str(uuid.uuid4())
        default_persona = "Auto-Transcriptionist"
        st.session_state.chats[default_id] = {
            "title": "Welcome Chat",
            "personality": default_persona,
            "messages": [
                {"role": "system", "content": PERSONALITIES[default_persona]["system_prompt"]},
                {"role": "assistant", "content": PERSONALITIES[default_persona]["greeting"]}
            ]
        }
        st.session_state.current_chat_id = default_id

    active_chat = st.session_state.chats[st.session_state.current_chat_id]
    active_persona = PERSONALITIES[active_chat["personality"]]

    # Page Header
    st.subheader(active_chat["personality"], anchor=False)
    st.caption(f"Scope: {active_persona['description']}")
    st.divider()

    # Chat render
    for msg in active_chat["messages"]:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -------------------------------------------------------------------------
    # Stream Response / Agent Step-by-Step Visualization
    # -------------------------------------------------------------------------
    if st.session_state.generate_response:
        with st.chat_message("assistant"):
            # Beautiful intermediate status display
            status_container = st.container()
            response_placeholder = st.empty()
            
            try:
                if not gemini_api_key or not serp_api_key:
                    st.error("Please configure both your Gemini and SerpApi keys in the sidebar expander.")
                    st.session_state.generate_response = False
                    st.stop()
                
                model_id = MODELS[st.session_state.global_model_name]
                
                with status_container:
                    status_widget = st.status("Agent initialized. Analysing scope...", expanded=True)
                
                # Retrieve execution stream generators
                agent_stream = run_agent_workflow(
                    chat_history=active_chat["messages"],
                    gemini_key=gemini_api_key,
                    serp_key=serp_api_key,
                    model_id=model_id
                )
                
                final_text = ""
                for step in agent_stream:
                    status = step.get("status")
                    message = step.get("message")
                    
                    if status == "thinking":
                        status_widget.update(label=f"🤔 {message}", state="running")
                    elif status == "searching":
                        status_widget.update(label=f"🔍 {message}", state="running")
                    elif status == "searching_done":
                        status_widget.write(f"✅ {message}")
                    elif status == "transcribing":
                        status_widget.update(label=f"🎙️ {message}", state="running")
                    elif status == "transcribing_done":
                        status_widget.write(f"✅ {message}")
                    elif status == "done":
                        status_widget.update(label="✨ Process Complete", state="complete", expanded=False)
                        final_text = message
                        response_placeholder.markdown(final_text)
                        
                active_chat["messages"].append({"role": "assistant", "content": final_text})
                
            except Exception as e:
                print(f"[Console API Error Exception]: {str(e)}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                st.error(f"Execution Error: {str(e)}")
            finally:
                st.session_state.generate_response = False
                st.rerun()

    # -------------------------------------------------------------------------
    # Bottom Layout: Model Selection & Native Input Box
    # -------------------------------------------------------------------------
    with st.bottom:
        user_input = st.chat_input("Ask Nexus to find or transcribe a video...")
        
        col_spacer, col_model = st.columns([0.70, 0.30])
        with col_spacer:
            st.write("")
            
        with col_model:
            with st.popover(st.session_state.global_model_name, key="model_popover", use_container_width=True):
                for model_name in MODELS.keys():
                    if st.button(model_name, key=f"m_opt_{model_name}", use_container_width=True):
                        st.session_state.global_model_name = model_name
                        st.rerun()

    # Handle Submission
    if user_input:
        query = user_input.strip()
        active_chat["messages"].append({"role": "user", "content": query})
        
        if len(active_chat["messages"]) <= 3:
            trimmed_title = query[:20] + ("..." if len(query) > 20 else "")
            active_chat["title"] = trimmed_title

        st.session_state.generate_response = True
        st.rerun()

if __name__ == "__main__":
    run_app()