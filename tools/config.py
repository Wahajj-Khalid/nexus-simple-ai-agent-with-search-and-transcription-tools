import os
import time

def configure_apis(gemini_key: str, serp_key: str):
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["SERPAPI_API_KEY"] = serp_key

def call_with_retry(func, *args, max_attempts=3, delay=2, **kwargs):
    attempt = 0
    while attempt < max_attempts:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "high demand" in error_str or "UNAVAILABLE" in error_str:
                attempt += 1
                if attempt >= max_attempts:
                    raise e
                time.sleep(delay * attempt)
            else:
                raise e