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
import logging, re, time, os
from logging.handlers import RotatingFileHandler


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
import os, json, time, logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

_DEFAULT_LOG_PATH = "./logs/llm_debug.jsonl"
_LLM_LOGGER = None


def _salvage_items_array(s: str) -> Optional[str]:
    i = s.find('"items"')
    if i < 0:
        return None
    j = s.find("[", i)
    if j < 0:
        return None

    k, depth, last_end = j + 1, 0, None
    in_str, esc = False, False
    while k < len(s):
        ch = s[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_end = k
        k += 1

    if last_end is None:
        return None
    arr = s[j : last_end + 1]  # [ {…}, {…}, … ] до последнего полного объекта
    return '{ "items": ' + arr + " }"


from collections.abc import Mapping


def _deep_get(d: Any, path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, Mapping):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            # допускаем объект с атрибутом .get
            try:
                cur = cur.get(part)  # type: ignore[attr-defined]
                if cur is None:
                    return default
            except Exception:
                return default
    return cur


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on", "y"}
    return False


def _as_str_path(v: Any, default: str) -> str:
    """Гарантируем строку-путь, иначе отдаём default."""
    if isinstance(v, str) and v.strip():
        return v
    return default


# ---- logger singleton
_LLM_LOGGER: Optional[logging.Logger] = None


def _get_llm_logger(path: str) -> logging.Logger:
    """path уже ДОЛЖЕН быть строкой — создаём ротационный JSONL-логгер."""
    global _LLM_LOGGER
    if _LLM_LOGGER:
        return _LLM_LOGGER
    dirn = os.path.dirname(path) or "."
    os.makedirs(dirn, exist_ok=True)
    logger = logging.getLogger("llmdebug")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    _LLM_LOGGER = logger
    return logger


def _redact_headers(h: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(h, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in h.items():
        if isinstance(v, str) and k.lower() in ("authorization", "proxy-authorization"):
            out[k] = "<REDACTED>"
        else:
            out[k] = v
    return out


def _log_llm_event(settings: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Безопасный JSONL-логгер: нормализуем типы, не падаем на ошибках."""
    try:
        # 1) включение через ENV имеет приоритет
        env_enabled = os.getenv("LLM_LOG_ENABLED")
        enabled = _as_bool(env_enabled) if env_enabled is not None else None

        # 2) иначе — из настроек
        if enabled is None:
            enabled = _as_bool(_deep_get(settings, "debug.llm_log_enabled", False))
        if not enabled:
            return

        # 3) путь: ENV -> settings -> default; жёстко приводим к str
        raw_path_env = os.getenv("LLM_LOG_PATH")
        raw_path_set = _deep_get(settings, "debug.llm_log_path", _DEFAULT_LOG_PATH)
        path = _as_str_path(raw_path_env if raw_path_env is not None else raw_path_set, _DEFAULT_LOG_PATH)

        event.setdefault("ts", time.time())
        _get_llm_logger(path).info(json.dumps(event, ensure_ascii=False))
    except Exception:
        # логирование не должно мешать рабочему потоку
        pass


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

    def _post_logged(
        self, *, url: str, headers: Dict[str, str], json_body: Optional[Dict[str, Any]] = None, form_body: Optional[Dict[str, Any]] = None, tag: str = "chat"
    ) -> requests.Response:
        t0 = time.perf_counter()
        r = self._session.post(url, headers=headers, json=json_body, data=form_body, timeout=self._timeout)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        try:
            resp_text = r.text
        except Exception:
            resp_text = None

        _log_llm_event(
            self.settings,
            {
                "provider": "gigachat",
                "phase": tag,  # "oauth" | "chat" | "chat_retry"
                "status": r.status_code,
                "duration_ms": dt_ms,
                "request": {
                    "method": "POST",
                    "url": url,
                    "headers": _redact_headers(headers),
                    "json": json_body,
                    "form": form_body,
                },
                "response": {
                    "headers": dict(r.headers or {}),
                    "text": resp_text,
                },
            },
        )
        return r

    # --- сервисные методы ---
    def _now(self) -> float:
        return time.time()

    def _invalidate_token(self) -> None:
        self._token = None
        self._exp_ts = 0.0

    def _ensure_token(self, force: bool = False) -> str:
        if not force and self._token and (self._now() < self._exp_ts - 60):
            return self._token

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {self._auth_key}",
            "RqUID": str(uuid.uuid4()),
        }
        data = {"scope": self._scope}
        try:
            resp = self._post_logged(url=self._auth_url, headers=headers, form_body=data, tag="oauth")
        except requests.exceptions.SSLError as e:
            raise RuntimeError("TLS verification failed for GigaChat OAuth. " "Set [llm.gigachat].verify_ssl to a PEM bundle path or false (temporary). " f"Details: {e}")
        if resp.status_code != 200:
            raise RuntimeError(f"GigaChat OAuth failed: {resp.status_code} {resp.text}")

        j = resp.json()
        token = j.get("access_token")
        exp = j.get("expires_at")
        if not token:
            raise RuntimeError("GigaChat OAuth: no access_token in response")
        if not isinstance(exp, (int, float)):
            exp = self._now() + 29 * 60

        self._token = token
        self._exp_ts = float(exp)
        return token

    # --- основной метод интерфейса ---
    def generate_text(self, prompt: str, opt: LLMOptions) -> Tuple[str, Dict[str, Any]]:
        msgs = [
            {"role": "system", "content": "You produce STRICT JSON only. No prose."},
            {"role": "user", "content": prompt},
        ]
        if getattr(opt, "log_prompt", False):
            _print_block("[PROMPT → GigaChat]", prompt, opt.truncate)

        def _build_headers(tok: str) -> Dict[str, str]:
            return {
                "Authorization": f"Bearer {tok}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Request-Source": "adaos-dsgen",
                "X-Client-Request-Id": str(uuid.uuid4()),
            }

        url = f"{self._base_url.rstrip('/')}/api/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": opt.model or "GigaChat",
            "messages": msgs,
        }
        if opt.max_tokens:
            body["max_tokens"] = int(opt.max_tokens)
        if opt.temperature is not None:
            body["temperature"] = float(opt.temperature)

        # 1-й вызов
        token = self._ensure_token()
        headers = _build_headers(token)
        r = self._post_logged(url=url, headers=headers, json_body=body, tag="chat")

        # если токен истёк — обновляем и повторяем ОДИН раз
        if r.status_code == 401 and "Token has expired" in (r.text or ""):
            _print_block("[GigaChat TOKEN]", "Access token expired → refresh & retry once", 500)
            self._invalidate_token()
            token = self._ensure_token(force=True)
            headers = _build_headers(token)
            r = self._post_logged(url=url, headers=headers, json_body=body, tag="chat_retry")

        # (опционально) общая обработка rate/5xx, если у тебя нет _post():
        if r.status_code in (429, 500, 502, 503, 504):
            # уважим Retry-After и сделаем один бэкофф-повтор
            ra = r.headers.get("Retry-After")
            try:
                wait = int(ra) if ra and str(ra).isdigit() else 2
            except Exception:
                wait = 2
            time.sleep(min(wait, 16))
            r = self._post_logged(url=url, headers=headers, json_body=body, tag="chat_retry2")

        if r.status_code != 200:
            raise RuntimeError(f"GigaChat chat error: {r.status_code} {r.text}")

        jr = r.json()
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
        """ use_responses = opt.use_responses and is_reasoning_model(opt.model)
        if use_responses and (opt.json_schema or opt.json_mode in ("schema", "object")):
            use_responses = False """
        use_responses = opt.use_responses and is_reasoning_model(opt.model)

        def _run(mode: str) -> Tuple[str, Dict[str, Any]]:
            if getattr(opt, "log_prompt", False):
                _print_block("[PROMPT → LLM]", prompt, opt.truncate)

            if use_responses:
                # ---------- Responses API: НЕ кладём response_format ----------
                p = {
                    "model": opt.model,
                    "input": [
                        {"role": "system", "content": msgs[0]["content"]},
                        {"role": "user", "content": msgs[1]["content"]},
                    ],
                    "max_output_tokens": int(opt.max_tokens),
                }
                if opt.reasoning_effort:
                    p["reasoning"] = {"effort": opt.reasoning_effort}

                # Усилим подсказку, если запрошен "schema"/"object"
                if mode in ("schema", "object"):
                    # мягкий инлайн-хинт вместо response_format
                    extra_sys = "Return a SINGLE valid JSON object only."
                    if mode == "schema" and opt.json_schema:
                        try:
                            name = opt.json_schema.get("json_schema", {}).get("name") or opt.json_schema.get("name") or "schema"
                            extra_sys += f" Conform to the '{name}' schema keys and types."
                        except Exception:
                            pass
                    p["input"][0]["content"] = p["input"][0]["content"] + " " + extra_sys

                raw_api = getattr(self._client.responses, "with_raw_response", None)
                t0 = time.perf_counter()
                try:
                    if raw_api and hasattr(raw_api, "create"):
                        raw = raw_api.create(**p)
                        resp = raw.parse()
                    else:
                        resp = self._client.responses.create(**p)
                except TypeError as e:
                    # старый SDK: подстраховка на случай, если где-то всё же проскочил response_format
                    if "response_format" in str(e):
                        # ретрай без любых «лишних» полей
                        p.pop("response_format", None)
                        if raw_api and hasattr(raw_api, "create"):
                            raw = raw_api.create(**p)
                            resp = raw.parse()
                        else:
                            t0 = time.perf_counter()
                            resp = self._client.responses.create(**p)
                    else:
                        raise

                # --- извлекаем текст из Responses ---
                text = getattr(resp, "output_text", "") or ""

                if not text:
                    # новая схема SDK: resp.output[0].content[0].text[0].data
                    try:
                        chunks = []
                        for out in getattr(resp, "output", None) or []:
                            for content in getattr(out, "content", None) or []:
                                if getattr(content, "type", None) == "output_text":
                                    text_obj = getattr(content, "text", None)
                                    # text_obj обычно list сегментов с .data
                                    if isinstance(text_obj, list):
                                        for seg in text_obj:
                                            chunk = getattr(seg, "data", None)
                                            if chunk:
                                                chunks.append(str(chunk))
                                    elif text_obj is not None:
                                        # на всякий случай: старый/иной формат
                                        chunks.append(str(text_obj))
                        text = "".join(chunks)
                    except Exception:
                        text = ""

                # жёсткий fallback: сериализуем весь resp, чтобы потом salvage вытащил "items"
                """ if not text:
                    try:
                        if hasattr(resp, "to_dict"):
                            text = json.dumps(resp.to_dict(), ensure_ascii=False)
                        elif hasattr(resp, "model_dump"):
                            text = json.dumps(resp.model_dump(), ensure_ascii=False)
                    except Exception:
                        text = "" """

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
                dt_ms = int((time.perf_counter() - t0) * 1000)
                _log_llm_event(
                    self.settings,
                    {
                        "provider": "openai",
                        "phase": "responses",
                        "mode": mode,
                        "model": opt.model,
                        "duration_ms": dt_ms,
                        "request": {"max_output_tokens": p.get("max_output_tokens")},
                        "response": {"meta": meta, "text": text},
                    },
                )

            else:
                # ---------- Chat Completions (как было) ----------
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
                    t0 = time.perf_counter()
                    raw = raw_api.create(**p)
                    resp = raw.parse()
                else:
                    t0 = time.perf_counter()
                    resp = self._client.chat.completions.create(**p)

                ch = resp.choices[0]
                msg = ch.message
                text = msg.content or ""
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
                dt_ms = int((time.perf_counter() - t0) * 1000)
                msg_info = {"has_content": bool(getattr(msg, "content", None)), "has_parsed": getattr(msg, "parsed", None) is not None}
                _log_llm_event(
                    self.settings,
                    {
                        "provider": "openai",
                        "phase": "chat",
                        "mode": mode,
                        "model": opt.model,
                        "phase": "chat",
                        "mode": mode,
                        "model": opt.model,
                        "duration_ms": dt_ms,
                        "request": {"max_completion_tokens": p.get("max_completion_tokens")},
                        "request": {"max_completion_tokens": p.get("max_completion_tokens")},
                        "response": {"meta": meta, "msg": msg_info},
                    },
                )
                if not text:
                    parsed = getattr(msg, "parsed", None)
                    if parsed is not None:
                        # превращаем обратно в строку JSON
                        try:
                            text = json.dumps(parsed, ensure_ascii=False)
                        except Exception:
                            text = str(parsed)
                if not text:
                    try:
                        # на случай другой формы объектов в SDK
                        text = getattr(msg, "to_dict", lambda: {})().get("content") or ""
                    except Exception:
                        pass
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
