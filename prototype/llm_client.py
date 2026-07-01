from __future__ import annotations

import json
import os
import re
import time
from typing import Optional
import openai
from dotenv import load_dotenv

from prototype import config

load_dotenv()


def chat(
    system: str,
    user: str,
    json_mode: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    retries: int = 4,
) -> str:
    api_key = os.environ.get(config.FORGE_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{config.FORGE_API_KEY_ENV} not set key")

    client = openai.OpenAI(api_key=api_key, base_url=config.FORGE_BASE_URL)
    kwargs = dict(
        model=model or config.FORGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"LLM retry {attempt + 1}/{retries} after: {exc} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_exc}")


def extract_json(text: str) -> Optional[dict | list]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            print(e)
    return None
