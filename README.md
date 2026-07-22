# Nexus

Nexus is an interactive AI Agent web application built using **Streamlit** and powered by **Gemini 3.5 Flash** and **Groq AI**. The platform allows users to find, transcribe, and reference YouTube videos and local video files in real time. The underlying agent uses function calling to search a local knowledge base, locate videos on YouTube via **SerpApi**, transcribe audio using **Groq Whisper API** or **Gemini Multimodal API**, and store the extracted transcripts as structured JSON records.

The codebase follows a modular structure that separates individual tool implementations, agent orchestration, styling, and user interface rendering.

---

## Live Deployment

The application is deployed on Streamlit Community Cloud and can be accessed directly at:
[Nexus](https://nexus-simple-ai-agent-with-search-and-transcription-tools.streamlit.app/)

---

## Repository Structure

```text
nexus-workspace/
├── .gitignore          # Specifies untracked files (venv, caches, secrets, temp audio)
├── README.md           # Project documentation and setup guide
├── backend.py          # Core agent reasoning loop and execution pipeline
├── frontend.py         # Streamlit user interface, state management, and routing
├── requirements.txt    # Python package dependencies
├── styles.css          # External CSS stylesheet for minimalist UI formatting
└── tools/              # Modular tool directory
    ├── __init__.py     # Package initialization and tool exports
    ├── config.py       # API key handlers and execution retry logic
    ├── knowledge_base.py # Local JSON database lookup and file creation
    ├── search.py       # SerpApi YouTube search implementation
    └── transcribe.py   # Groq Whisper and Gemini transcription handlers
```

---

## Technical Features

* **Multi-Tool Agent Function Calling**: The agent dynamically selects and executes tools based on user input, handling search, retrieval, and transcription workflows autonomously.
* **Local JSON Knowledge Base**: Automatically checks an internal `knowledge_base/` directory for pre-existing transcripts before initiating web searches. Transcripts are stored as structured JSON files containing titles, URLs, and text content.
* **Multi-Engine Audio Transcription**: Utilizes Groq's high-speed `whisper-large-v3` API as the primary audio transcription engine, with Gemini Multimodal as an automatic fallback.
* **Local Video File Uploads**: Includes an upward-expanding popover interface enabling users to upload and transcribe local video files directly.
* **Resilient Rate-Limit Fallbacks**: Incorporates exponential backoff retries and cooldown delays to prevent API rate-limit interruptions.
* **Sleek, Minimalist UI**: Styled via an external `styles.css` file with flat-text layouts, custom list structures, thread management, and no unnecessary UI clutter.

---

## Local Setup and Installation

Follow these instructions to get a local copy of the application running.

### Prerequisites

* **Python 3.9 or higher** installed on your machine.
* A valid **Gemini API key** from [Google AI Studio](https://aistudio.google.com/).
* A valid **SerpApi key** from [SerpApi](https://serpapi.com/).
* An optional **Groq API key** from the [Groq Console](https://console.groq.com/) for faster Whisper transcriptions.

### 1. Clone the Repository

Clone this workspace to your local directory:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Configure a Virtual Environment

Create and activate a Python virtual environment:

**On macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

You can configure API credentials in one of two ways:

* **Option A: Local Secrets Configuration (Recommended)**
  Create a `.streamlit` folder at the root of the project, create a `secrets.toml` file inside it, and add your API keys:

  ```toml
  GEMINI_API_KEY = "your_actual_gemini_api_key_here"
  SERPAPI_API_KEY = "your_actual_serpapi_api_key_here"
  GROQ_API_KEY = "your_actual_groq_api_key_here"
  ```

* **Option B: UI Settings Panel**
  If no secrets file is present, the application provides input fields inside the sidebar's Settings panel to enter the credentials manually.

---

## Running the Application Locally

Start the local Streamlit server by executing:

```bash
streamlit run frontend.py
```

If the browser window does not open automatically, copy and paste the local network URL (typically `http://localhost:8501`) provided in your terminal output.