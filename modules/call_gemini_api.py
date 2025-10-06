import requests
import json
import time
from typing import List, Dict
import os

def call_gemini_api(user_prompt: str, system_prompt: str = "", tools: List[Dict] = None) -> str:
    API_KEY = os.environ.get("GEMINI_API_KEY")
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    MODEL_NAME = "gemini-2.5-flash-preview-05-20"

    """Handles the API call to Gemini with exponential backoff."""
    url = f"{API_BASE_URL}/{MODEL_NAME}:generateContent?key={API_KEY}"

    # 1. Prepare the payload
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": user_prompt}
                ]
            }
        ]
    }

    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [
                {"text": system_prompt}
            ]
        }

    if tools:
        payload["tools"] = tools

    # 2. Execute with exponential backoff (e.g., 1s, 2s, 4s, 8s)
    max_retries = 4
    for attempt in range(max_retries):
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()

            # Extract text
            candidate = result.get('candidates', [{}])[0]
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')

            if text:
                return text

            return "AI returned no content."

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429 and attempt < max_retries - 1:
                # Handle rate limiting with backoff
                sleep_time = 2 ** attempt
                print(f"Rate limited (429). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                # Other HTTP errors or last rate-limit attempt
                return f"HTTP Error: {http_err} - {response.text}"
        except Exception as err:
            return f"An error occurred during API call: {err}"

    return "Failed to get a response after multiple retries."
