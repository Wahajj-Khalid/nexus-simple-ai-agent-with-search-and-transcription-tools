import os
import re
import json

STOP_WORDS = {
    "find", "search", "video", "transcribe", "transcript", "transcription",
    "explaining", "explain", "about", "how", "to", "for", "free", "audio",
    "and", "the", "a", "an", "in", "seconds", "100", "short", "get", "me",
    "please", "show", "with", "it", "of", "on", "youtube", "link", "text"
}

def search_knowledge_base(query: str) -> dict:
    kb_dir = "knowledge_base"
    
    if not os.path.exists(kb_dir):
        return {"status": "not_found", "message": "Knowledge base directory does not exist yet."}
        
    raw_tokens = [t.lower() for t in re.findall(r'\w+', query)]
    
    # Filter out generic stop-words to isolate subject keywords
    topic_tokens = [t for t in raw_tokens if t not in STOP_WORDS and len(t) > 1]
    
    if not topic_tokens:
        return {"status": "not_found", "message": "No specific topic keywords found in query."}
        
    best_match = None
    best_score = 0
    
    for filename in os.listdir(kb_dir):
        if filename.endswith(".json"):
            fn_lower = filename.lower()
            
            # Count matches only for actual topic keywords
            score = sum(1 for token in topic_tokens if token in fn_lower)
            if score > best_score:
                best_score = score
                best_match = filename
                
    # Strictly require that all topic tokens match the filename
    if best_match and best_score >= len(topic_tokens):
        file_path = os.path.join(kb_dir, best_match)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "status": "success",
                "title": data.get("title", ""),
                "source_url": data.get("source_url", ""),
                "transcript": data.get("transcript", "")
            }
        except Exception as e:
            return {"status": "error", "message": f"Error reading JSON record: {str(e)}"}
            
    return {"status": "not_found", "message": "No matching topic found in knowledge base."}

def write_to_knowledge_base(title: str, source_url: str, transcript_text: str) -> str:
    kb_dir = "knowledge_base"
    os.makedirs(kb_dir, exist_ok=True)
    safe_title = re.sub(r'[^\w\-_\. ]', '_', title)
    kb_path = os.path.join(kb_dir, f"{safe_title}.json")
    
    data = {
        "title": title,
        "source_url": source_url,
        "transcript": transcript_text
    }
    
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    return kb_path