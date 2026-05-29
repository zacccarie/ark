"""Phase 65 — LLM provider abstraction.

Supporte OpenAI, Anthropic, OpenRouter, Mistral, Ollama (local).
Tous via httpx. Pas de SDK lourd. Streaming SSE supporté.

Secrets stockés ~/.ark/secrets.json :
{
  "openai": {"api_key": "sk-..."},
  "anthropic": {"api_key": "sk-ant-..."},
  "openrouter": {"api_key": "sk-or-..."},
  "ollama": {"base_url": "http://localhost:11434"}
}
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Optional


SECRETS_PATH = Path.home() / ".ark" / "secrets.json"


def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        return {}
    try:
        return json.loads(SECRETS_PATH.read_text())
    except Exception:
        return {}


def save_secrets(d: dict) -> None:
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_PATH.write_text(json.dumps(d, indent=2))
    try:
        SECRETS_PATH.chmod(0o600)
    except Exception:
        pass


# ============================================================
# Provider interface
# ============================================================

class LLMProvider:
    """Unified messages list:
    [
      {"role": "system" | "user" | "assistant", "content": "..."},
    ]
    """
    name: str = "base"

    def chat(self, messages: list[dict], model: str,
             max_tokens: int = 2048, temperature: float = 0.7,
             stream: bool = False) -> Any:
        raise NotImplementedError


# ============================================================
# OpenAI
# ============================================================

class OpenAIProvider(LLMProvider):
    name = "openai"
    base_url = "https://api.openai.com/v1"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        if base_url:
            self.base_url = base_url.rstrip("/")

    def chat(self, messages, model="gpt-4o-mini",
             max_tokens=2048, temperature=0.7, stream=False):
        import httpx
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            return self._stream(url, payload, headers)
        r = httpx.post(url, json=payload, headers=headers, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _stream(self, url, payload, headers):
        import httpx
        with httpx.stream("POST", url, json=payload,
                              headers=headers, timeout=300.0) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


# ============================================================
# Anthropic
# ============================================================

class AnthropicProvider(LLMProvider):
    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(self, messages, model="claude-3-5-sonnet-latest",
             max_tokens=2048, temperature=0.7, stream=False):
        import httpx
        # Anthropic distinguishes system from messages
        system = ""
        msgs = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                msgs.append(m)
        payload = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system.strip():
            payload["system"] = system.strip()
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/messages"
        if stream:
            return self._stream(url, payload, headers)
        r = httpx.post(url, json=payload, headers=headers, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        return data["content"][0]["text"]

    def _stream(self, url, payload, headers):
        import httpx
        with httpx.stream("POST", url, json=payload,
                              headers=headers, timeout=300.0) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[6:]
                try:
                    obj = json.loads(chunk)
                    if obj.get("type") == "content_block_delta":
                        delta = obj.get("delta", {}).get("text", "")
                        if delta:
                            yield delta
                    elif obj.get("type") == "message_stop":
                        break
                except Exception:
                    continue


# ============================================================
# OpenRouter (proxy multi-model)
# ============================================================

class OpenRouterProvider(OpenAIProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"


# ============================================================
# Mistral
# ============================================================

class MistralProvider(OpenAIProvider):
    name = "mistral"
    base_url = "https://api.mistral.ai/v1"


# ============================================================
# Ollama (local)
# ============================================================

class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def chat(self, messages, model="llama3.2",
             max_tokens=2048, temperature=0.7, stream=False):
        import httpx
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stream:
            return self._stream(url, payload)
        r = httpx.post(url, json=payload, timeout=300.0)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]

    def _stream(self, url, payload):
        import httpx
        with httpx.stream("POST", url, json=payload, timeout=600.0) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("done"):
                        break
                    delta = obj.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


# ============================================================
# Factory
# ============================================================

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "openrouter": "anthropic/claude-3.5-sonnet",
    "mistral": "mistral-large-latest",
    "ollama": "llama3.2",
}


def get_provider(name: str) -> LLMProvider:
    secrets = load_secrets()
    cfg = secrets.get(name, {})
    if name == "openai":
        return OpenAIProvider(api_key=cfg.get("api_key", ""))
    if name == "anthropic":
        return AnthropicProvider(api_key=cfg.get("api_key", ""))
    if name == "openrouter":
        return OpenRouterProvider(api_key=cfg.get("api_key", ""))
    if name == "mistral":
        return MistralProvider(api_key=cfg.get("api_key", ""))
    if name == "ollama":
        return OllamaProvider(base_url=cfg.get(
            "base_url", "http://localhost:11434"))
    raise ValueError(f"Unknown provider: {name}")


def available_providers() -> list[str]:
    return ["anthropic", "openai", "openrouter", "mistral", "ollama"]
