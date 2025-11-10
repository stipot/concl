# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol

from settings_loader import load_settings, get_typed


# ---------- Интерфейс ----------
@dataclass
class LLMOptions:
    model: str
    max_tokens: int = 3000
    temperature: Optional[float] = None
    seed: Optional[int] = None
    json_schema: Optional[Dict[str, Any]] = None
    json_mode: str = "schema"  # "schema" | "object" | "none"
    use_responses: bool = False  # только для официального OpenAI
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"
    log_prompt: bool = False
    log_response: bool = False
    truncate: int = 2000


class LLMProvider(Protocol):
    def generate_text(self, prompt: str, opt: LLMOptions) -> Tuple[str, Dict[str, Any]]: ...


REASONING_HINTS = ("gpt-5", "o3", "o4")


def is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return any(h in n for h in REASONING_HINTS)


def _print_block(title: str, text: str, max_len: int = 2000):
    body = text or ""
    if len(body) > max_len:
        body = body[:max_len] + "\n…[truncated]"
    print(f"\n{title}\n{body}\n" + "-" * 60)


# ---------- Базовый конструктор OpenAI клиента с фильтрацией типов ----------
def _build_openai_client(
    api_key: str,
    base_url: Optional[str],
    timeout: Optional[float],
    max_retries: Optional[int],
    default_headers: Optional[Dict[str, str]],
    default_query: Optional[Dict[str, object]],
):
    from openai import OpenAI

    kw: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kw["base_url"] = str(base_url).rstrip("/")
    if isinstance(timeout, (int, float)):
        kw["timeout"] = float(timeout)
    if isinstance(max_retries, int):
        kw["max_retries"] = max_retries
    if isinstance(default_headers, dict):
        kw["default_headers"] = default_headers
    if isinstance(default_query, dict):
        kw["default_query"] = default_query
    return OpenAI(**kw)


# ---------- Реализации ----------
class OpenAIProvider(LLMProvider):
    def __init__(self, settings=None):
        # читаем настройки
        self.settings = settings or load_settings()
        api_key = get_typed(self.settings, "OPENAI_API_KEY", None, str)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .secrets.toml")
        base_url = get_typed(self.settings, "llm.openai.base_url", None, str)
        timeout = get_typed(self.settings, "llm.openai.timeout", None, float)
        max_retries = get_typed(self.settings, "llm.openai.max_retries", None, int)
        default_headers = get_typed(self.settings, "llm.openai.default_headers", None, dict)
        default_query = get_typed(self.settings, "llm.openai.default_query", None, dict)
        self._client = _build_openai_client(api_key, base_url, timeout, max_retries, default_headers, default_query)

    def generate_text(self, prompt: str, opt: LLMOptions) -> Tuple[str, Dict[str, Any]]:
        msgs = [
            {"role": "system", "content": "You produce STRICT JSON only. No prose."},
            {"role": "user", "content": prompt},
        ]
        use_responses = opt.use_responses and is_reasoning_model(opt.model)

        def _run(mode: str) -> Tuple[str, Dict[str, Any]]:
            if getattr(opt, "log_prompt", False):
                _print_block("[PROMPT → LLM]", prompt, opt.truncate)
            if use_responses:
                p = {
                    "model": opt.model,
                    "input": [
                        {"role": "system", "content": msgs[0]["content"]},
                        {"role": "user", "content": msgs[1]["content"]},
                    ],
                    "max_output_tokens": int(opt.max_tokens),
                }
                if mode == "schema" and opt.json_schema:
                    p["response_format"] = {"type": "json_schema", "json_schema": opt.json_schema}
                elif mode == "object":
                    p["response_format"] = {"type": "json_object"}
                if opt.reasoning_effort:
                    p["reasoning"] = {"effort": opt.reasoning_effort}

                raw_api = getattr(self._client.responses, "with_raw_response", None)
                if raw_api and hasattr(raw_api, "create"):
                    raw = raw_api.create(**p)
                    resp = raw.parse()
                else:
                    resp = self._client.responses.create(**p)

                text = getattr(resp, "output_text", "") or ""
                if not text:
                    try:
                        for out in getattr(resp, "output", None) or []:
                            for part in getattr(out, "content", None) or []:
                                if getattr(part, "type", None) == "output_text":
                                    text += part.text or ""
                    except Exception:
                        pass
                finish = getattr(resp, "finish_reason", None)
                usage = getattr(resp, "usage", None)
                meta = {
                    "id": getattr(resp, "id", None),
                    "model": opt.model,
                    "finish_reason": finish,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "completion_tokens": getattr(usage, "output_tokens", None) if usage else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                }
            else:
                p = {
                    "model": opt.model,
                    "messages": msgs,
                    "max_completion_tokens": int(opt.max_tokens),
                }
                if opt.temperature is not None and float(opt.temperature) != 1.0:
                    p["temperature"] = float(opt.temperature)
                if opt.seed is not None:
                    p["seed"] = int(opt.seed)
                if mode == "schema" and opt.json_schema:
                    p["response_format"] = {"type": "json_schema", "json_schema": opt.json_schema}
                elif mode == "object":
                    p["response_format"] = {"type": "json_object"}

                raw_api = getattr(self._client.chat.completions, "with_raw_response", None)
                if raw_api and hasattr(raw_api, "create"):
                    raw = raw_api.create(**p)
                    resp = raw.parse()
                else:
                    resp = self._client.chat.completions.create(**p)

                ch = resp.choices[0]
                text = ch.message.content or ""
                finish = getattr(ch, "finish_reason", None)
                usage = getattr(resp, "usage", None)
                meta = {
                    "id": getattr(resp, "id", None),
                    "model": getattr(resp, "model", opt.model),
                    "finish_reason": finish,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                }

            if not (text or "").strip():
                raise ValueError(f"Empty content (finish_reason={meta.get('finish_reason')})")
            if opt.log_response:
                _print_block(f"[LLM RESPONSE] {meta}", text, opt.truncate)
            return text, meta

        try:
            return _run("schema")
        except Exception:
            try:
                return _run("object")
            except Exception:
                msgs.append({"role": "system", "content": 'Answer MUST be a single JSON object with key "items" only, ≤1200 tokens.'})
                return _run("none")


class OpenAICompatProvider(OpenAIProvider):
    """OpenAI-совместимые API (например, GigaChat Pro): всегда Chat Completions"""

    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        api_key = get_typed(self.settings, "GIGACHAT_API_KEY", None, str) or get_typed(self.settings, "OPENAI_API_KEY", None, str)
        if not api_key:
            raise RuntimeError("GIGACHAT_API_KEY (или OPENAI_API_KEY) is not set in .secrets.toml")
        base_url = get_typed(self.settings, "llm.compat.base_url", None, str)
        timeout = get_typed(self.settings, "llm.compat.timeout", None, float)
        max_retries = get_typed(self.settings, "llm.compat.max_retries", None, int)
        default_headers = get_typed(self.settings, "llm.compat.default_headers", None, dict)
        default_query = get_typed(self.settings, "llm.compat.default_query", None, dict)
        self._client = _build_openai_client(api_key, base_url, timeout, max_retries, default_headers, default_query)

    def generate_text(self, prompt: str, opt: LLMOptions) -> Tuple[str, Dict[str, Any]]:
        opt.use_responses = False
        opt.reasoning_effort = None
        return super().generate_text(prompt, opt)
