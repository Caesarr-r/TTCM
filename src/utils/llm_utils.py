import os
import torch
import numpy as np
import random
import json
import requests
import re
import time
from typing import List, Dict, Optional, Any
from requests.exceptions import Timeout, RequestException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- LLM API Configuration ---
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_HEADERS = {
    "Authorization": "your_api_key",
    "Content-Type": "application/json"
}
BASE_PAYLOAD = {
    "model": "Pro/deepseek-ai/DeepSeek-V3",
    "max_tokens": 4096,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "temperature": 0.3,
    "top_p": 0.6,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1
}

def call_llm_api(messages: List[dict], response_format: Optional[Dict] = None, timeout: int = 300, retries: int = 3, backoff_factor: float = 2.0, debug: bool = False) -> str:
    """
    """
    payload = BASE_PAYLOAD.copy()
    payload["messages"] = messages
    if response_format:
        payload["response_format"] = response_format

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=API_HEADERS, json=payload, timeout=timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()

            if response_json.get("choices") and len(response_json["choices"]) > 0:
                llm_content = response_json["choices"][0].get("message", {}).get("content", "").strip()
                return llm_content
            else:
                return ""
        except Timeout:
            if attempt == retries - 1:
                return "LLM_API_TIMEOUT_ERROR"
        except RequestException as e:
            if attempt == retries - 1:
                return "LLM_API_REQUEST_ERROR"
        except Exception as e:
            if attempt == retries - 1:
                return "LLM_API_UNKNOWN_ERROR"

        if attempt < retries - 1:
            sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)

    return ""
