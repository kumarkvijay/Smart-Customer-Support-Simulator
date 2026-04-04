from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request


@dataclass(frozen=True)
class OllamaResult:
    ok: bool
    content: str
    error: str = ""


def chat_with_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int = 45,
) -> OllamaResult:
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 256,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            return OllamaResult(False, "", "Ollama returned an empty response.")
        return OllamaResult(True, content)
    except urllib.error.URLError as exc:
        return OllamaResult(False, "", f"Ollama connection failed: {exc}")
    except Exception as exc:
        return OllamaResult(False, "", f"Unexpected Ollama error: {exc}")

