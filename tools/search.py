import os
import requests

def search_video(query: str) -> str:
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