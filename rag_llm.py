"""
rag_llm.py – Thin wrapper that calls a local Ollama‑style HTTP endpoint
instead of loading the model with `transformers`.

Environment variables you can override:
  • LLM_ENDPOINT   – base URL of the server (default: http://host.docker.internal:11434)
  • LLM_MODEL      – model name at that endpoint   (default: mistral)
"""

import os
import requests

ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:11434")
MODEL_ID = os.getenv("LLM_MODEL", "mistral")
URL = f"{ENDPOINT.rstrip('/')}/api/generate"


def _post(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Internal helper that sends the JSON payload and returns raw server JSON."""
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }
    resp = requests.post(URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()  # expected to contain a "response" key


def ask(prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Send `prompt` to the HTTP LLM endpoint and return the generated text.

    Raises `requests.HTTPError` if the server returns an error.
    """
    data = _post(prompt, max_tokens=max_tokens, temperature=temperature)
    # Ollama‑style servers return the completion in the "response" field
    return data.get("response", "").strip()
