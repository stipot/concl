# step01\llm_provider.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol, cast, Union

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from settings_loader import load_settings, get_typed
import time, uuid, base64
import requests


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
# ---------- GigaChat (native REST) ----------
import time, uuid, base64
import requests


class GigaChatProvider(LLMProvider):
    """
    Нативный клиент GigaChat REST API.
    Требует ключ авторизации (Basic) для получения OAuth access_token,
    далее использует Bearer токен для /api/v1/chat/completions.
    """

    def __init__(self, settings=None):
        self.settings = settings or load_settings()

        self._auth_url = get_typed(self.settings, "llm.gigachat.auth_url", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth", str)
        self._base_url = get_typed(self.settings, "llm.gigachat.base_url", "https://gigachat.devices.sberbank.ru", str)
        self._scope = get_typed(self.settings, "llm.gigachat.scope", "GIGACHAT_API_PERS", str)

        # --- timeout: берём как object -> приводим к float с подсказкой типов ---
        timeout_raw = cast(Union[int, float, str], get_typed(self.settings, "llm.gigachat.timeout", 100.0, object))
        try:
            self._timeout = float(timeout_raw)
        except (TypeError, ValueError):
            self._timeout = 100.0
        print("self._timeout", self._timeout)
        # --- auth key ---
        auth_key = get_typed(self.settings, "GIGACHAT_AUTH_KEY", None, str)
        if not auth_key:
            cid = get_typed(self.settings, "GIGACHAT_CLIENT_ID", None, str)
            csec = get_typed(self.settings, "GIGACHAT_CLIENT_SECRET", None, str)
            if not (cid and csec):
                raise RuntimeError("Set GIGACHAT_AUTH_KEY or GIGACHAT_CLIENT_ID/CLIENT_SECRET in .secrets.toml")
            auth_key = base64.b64encode(f"{cid}:{csec}".encode("utf-8")).decode("ascii")
        self._auth_key = auth_key

        # --- verify_ssl: путь/True/False c явной типизацией ---
        verify_raw = cast(Union[bool, int, str], get_typed(self.settings, "llm.gigachat.verify_ssl", False, object))
        if isinstance(verify_raw, str) and verify_raw.strip():
            self._verify = verify_raw.strip()  # путь к PEM bundle
        else:
            self._verify = bool(verify_raw)  # True / False

        # --- Session с ретраями (без дублей) ---
        self._session = Session()
        self._session.verify = self._verify
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["HEAD", "GET", "OPTIONS", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=1, pool_maxsize=20)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # --- кэш токена ---
        self._token: Optional[str] = None
        self._exp_ts: float = 0.0
        """ self._session = requests.Session()
        if isinstance(verify_cfg_raw, str) and verify_cfg_raw.strip():
            # путь к кастомному bundle PEM
            self._session.verify = verify_cfg_raw.strip()
        else:
            # True/False
            self._session.verify = bool(verify_cfg_raw)
        self._session.verify = self._verify
        timeout_raw = get_typed(self.settings, "llm.gigachat.timeout", 40.0, object)
        try:
            self._timeout = float(timeout_raw)
        except (TypeError, ValueError):
            self._timeout = 40.0 """

    # --- сервисные методы ---
    def _now(self) -> float:
        return time.time()

    def _ensure_token(self) -> str:
        # обновляем за минуту до истечения
        if self._token and (self._now() < self._exp_ts - 60):
            return self._token

        # запрос токена по OAuth 2.0
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {self._auth_key}",
            "RqUID": str(uuid.uuid4()),
        }
        data = {"scope": self._scope}
        print("step_insure_token")
        try:
            resp = self._session.post(self._auth_url, headers=headers, data=data, timeout=self._timeout)
        except requests.exceptions.SSLError as e:
            raise RuntimeError("TLS verification failed for GigaChat OAuth. " "Set [llm.gigachat].verify_ssl to a PEM bundle path or false (temporary). " f"Details: {e}")
        if resp.status_code != 200:
            raise RuntimeError(f"GigaChat OAuth failed: {resp.status_code} {resp.text}")

        j = resp.json()
        token = j.get("access_token")
        exp = j.get("expires_at")  # unix ts по докам
        if not token:
            raise RuntimeError("GigaChat OAuth: no access_token in response")
        # если expires_at нет, дадим дефолт 29 мин
        if not isinstance(exp, (int, float)):
            exp = self._now() + 29 * 60

        self._token = token
        self._exp_ts = float(exp)
        return token

    # --- основной метод интерфейса ---
    def generate_text(self, prompt: str, opt: LLMOptions) -> Tuple[str, Dict[str, Any]]:
        # Готовим сообщения (как у OpenAI)
        msgs = [
            {"role": "system", "content": "You produce STRICT JSON only. No prose."},
            {"role": "user", "content": prompt},
        ]
        if getattr(opt, "log_prompt", False):
            _print_block("[PROMPT → GigaChat]", prompt, opt.truncate)

        token = self._ensure_token()

        url = f"{self._base_url.rstrip('/')}/api/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": opt.model or "GigaChat",
            "messages": msgs,
        }
        # стандартные сэмплинг-параметры. Если какие-то не поддерживаются, сервер их проигнорирует.
        if opt.max_tokens:
            body["max_tokens"] = int(opt.max_tokens)
        if opt.temperature is not None:
            body["temperature"] = float(opt.temperature)

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        r = self._session.post(url, headers=headers, json=body, timeout=self._timeout)
        if r.status_code != 200:
            raise RuntimeError(f"GigaChat chat error: {r.status_code} {r.text}")

        jr = r.json()
        # формат ответа схож с OpenAI: choices[0].message.content
        try:
            ch0 = (jr.get("choices") or [])[0]
            text = ((ch0.get("message") or {}).get("content") or "").strip()
            finish = ch0.get("finish_reason")
        except Exception:
            text = ""
            finish = None

        usage = jr.get("usage") or {}
        meta = {
            "id": jr.get("id"),
            "model": jr.get("model") or (opt.model or "GigaChat"),
            "finish_reason": finish,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

        if not text:
            raise ValueError(f"GigaChat returned empty content (finish_reason={finish})")

        if getattr(opt, "log_response", False):
            _print_block(f"[GigaChat RESPONSE] {meta}", text, opt.truncate)

        return text, meta


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
