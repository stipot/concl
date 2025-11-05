# qgen_cli.py
# Генерация контекстно-зависимых вопросов с Typer-CLI.
# Требуется: pip install typer[all] toml openai
from __future__ import annotations

import json
import os
import re
import time
import random
import datetime as dt
from typing import Any, Optional, Dict, List, Tuple

import typer
import toml

# --------- Константы / пути ---------
app = typer.Typer(add_completion=False, no_args_is_help=True)
DATA_DIR = "./step01/data"
SECRETS = ".secrets.toml"
DEFAULT_OUT_TMPL = "{model}_generated_questions_{lang}.jsonl"
SUPPORTED_MODELS = [
    "gpt-5",  # placeholder для будущих настроек
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
]

# --------- Утилиты ---------
WORD_RE = re.compile(r"\w+", flags=re.U | re.M)


def qa_batch_schema(expected_max: int | None = None) -> Dict[str, Any]:
    """Топ-уровень — object c полем items: array[{q,n,v[{c,a}]}]."""
    qa_item = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "minLength": 3},
            "n": {"type": "string", "minLength": 1},
            "v": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"c": {"type": "string", "minLength": 3}, "a": {"type": "string", "minLength": 1}},
                    "required": ["c", "a"],
                    "additionalProperties": False,
                },
                "minItems": 2,
                "maxItems": 4,
            },
        },
        "required": ["q", "n", "v"],
        "additionalProperties": False,
    }

    schema: Dict[str, Any] = {"type": "object", "properties": {"items": {"type": "array", "items": qa_item, "minItems": 1}}, "required": ["items"], "additionalProperties": False}
    if isinstance(expected_max, int) and expected_max > 0:
        schema["properties"]["items"]["maxItems"] = expected_max

    return {"name": "qa_batch", "schema": schema, "strict": True}


def read_api_key(path: str = SECRETS) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        secrets = toml.load(path)
        return secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def load_field_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_words(s: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer((s or "").strip())]


def now_iso() -> str:
    return dt.datetime.now().isoformat()


def extract_json_block(reply: str) -> str:
    """
    Пытается вернуть корректный JSON (объект или массив) из ответа модели.
    1) снимает ``` и лишний текст вокруг;
    2) если строка начинается с { ... } или [ ... ] — НЕ ищет внутренние массивы;
    3) если обнаружен одиночный объект — вернёт его как есть (объект),
       парсер затем сам обернёт в список при необходимости.
    """
    s = (reply or "").strip()

    # снять ограждения ```json ... ``` / ``` ... ```
    if s.startswith("```json"):
        s = s[7:]
        if s.endswith("```"):
            s = s[:-3]
    elif s.startswith("```"):
        s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
    s = s.strip()

    # если уже объект или массив — вернём как есть (без поиска внутренних [])
    if s.startswith("{") and s.endswith("}"):
        return s
    if s.startswith("[") and s.endswith("]"):
        return s

    # попытка аккуратно вырезать верхнеуровневый объект по балансировке скобок
    def _extract_top_level_obj(txt: str) -> Optional[str]:
        depth = 0
        start = None
        for i, ch in enumerate(txt):
            if ch == "{":
                if start is None:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return txt[start : i + 1]
        return None

    def _extract_top_level_arr(txt: str) -> Optional[str]:
        depth = 0
        start = None
        for i, ch in enumerate(txt):
            if ch == "[":
                if start is None:
                    start = i
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and start is not None:
                    return txt[start : i + 1]
        return None

    obj = _extract_top_level_obj(s)
    if obj:
        return obj
    arr = _extract_top_level_arr(s)
    if arr:
        return arr

    # ничего уверенного — возвращаем как есть (позже fallback попытается разобрать кусочно)
    return s


def _create_with_raw_response(client, **params):
    """
    Пытается вызвать chat.completions.with_raw_response.create(...)
    и вернуть (parsed_resp, raw_status, raw_headers, raw_text).
    Если .with_raw_response недоступен — падает обратно на обычный create,
    возвращая raw_* как None.
    """
    try:
        raw_api = getattr(client.chat.completions, "with_raw_response", None)
        if raw_api and hasattr(raw_api, "create"):
            raw = raw_api.create(**params)
            # raw.* доступен до parse()
            status = getattr(raw, "status_code", None)
            headers = dict(getattr(raw, "headers", {}) or {})
            try:
                raw_text = raw.text  # тело целиком как строка
            except Exception:
                raw_text = None
            # Парсим в привычный объект SDK
            resp = raw.parse()
            return resp, status, headers, raw_text
    except Exception as e:
        # если сырой режим не удался — логнем и пойдём обычным путём
        print_block("[RAW HTTP ERROR]", f"{e}", max_len=2000, color=typer.colors.RED)

    # fallback: обычный вызов, без сырого тела
    resp = client.chat.completions.create(**params)
    return resp, None, None, None


def _dump_choice_debug(resp):
    try:
        ch = resp.choices[0]
        msg = ch.message
        meta_lines = [
            f"id={getattr(resp, 'id', None)}",
            f"model={getattr(resp, 'model', None)}",
            f"finish_reason={getattr(ch, 'finish_reason', None)}",
            f"role={getattr(msg, 'role', None)}",
            f"has_tool_calls={bool(getattr(msg, 'tool_calls', None))}",
            f"has_refusal={bool(getattr(msg, 'refusal', None))}",
            f"content_len={len(msg.content or '')}",
        ]
        # usage может отсутствовать
        try:
            usage = resp.usage
            if usage:
                meta_lines.append(f"usage.prompt_tokens={getattr(usage, 'prompt_tokens', None)}")
                meta_lines.append(f"usage.completion_tokens={getattr(usage, 'completion_tokens', None)}")
                meta_lines.append(f"usage.total_tokens={getattr(usage, 'total_tokens', None)}")
        except Exception:
            pass

        # выведем refusal/tool_calls если есть
        extras = []
        if getattr(msg, "refusal", None):
            extras.append(f"refusal={msg.refusal}")
        if getattr(msg, "tool_calls", None):
            extras.append(f"tool_calls={msg.tool_calls}")
        debug_text = "\n".join(meta_lines + extras)

        print_block("[GPT META]", debug_text, max_len=5000, color=typer.colors.YELLOW)
    except Exception as e:
        print_block("[GPT META ERROR]", f"{e}", max_len=1000, color=typer.colors.RED)


def print_block(title: str, text: str, max_len: int = 2000, color=typer.colors.BRIGHT_BLACK):
    body = text or ""
    trunc_note = ""
    if len(body) > max_len:
        trunc_note = f" [truncated {len(body) - max_len} chars]"
        body = body[:max_len]
    typer.secho(f"\n{title}{trunc_note}", fg=color)
    typer.echo(body)
    typer.secho("-" * 60, fg=color)


def parse_json_items(reply: str) -> List[Dict]:
    s = extract_json_block(reply)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []

    # Корневой объект с массивом items (Structured Outputs)
    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
        return [x for x in obj["items"] if isinstance(x, dict)]

    # Одиночный объект (допускаем, но это не должно случаться при json_schema)
    if isinstance(obj, dict):
        return [obj]

    # Уже массив объектов
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    return []


def load_processed_subjects(output_path: str) -> set[Tuple[str, str, str]]:
    done: set[Tuple[str, str, str]] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = (str(obj.get("f", "")).strip(), str(obj.get("s", "")).strip(), str(obj.get("j", "")).strip())
            if all(key):
                done.add(key)
    return done


def write_jsonl(path: str, items: List[Dict]) -> None:
    # безопасно создаём папку только если она непуста
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    abs_path = os.path.abspath(path)
    with open(abs_path, "a", encoding="utf-8") as f:
        for it in items:
            line = json.dumps(it, ensure_ascii=False)
            f.write(line + "\n")
    typer.secho(f"[FILE] +{len(items)} lines → {abs_path}", fg=typer.colors.BLUE)


@app.callback(invoke_without_command=True)
def _print_cwd(ctx: typer.Context):
    # Печатаем рабочую директорию один раз, это помогает ловить относительные пути
    typer.secho(f"[CWD] {os.path.abspath(os.getcwd())}", fg=typer.colors.BRIGHT_BLACK)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# --------- Мини-валидация (локальная, до большой валидации) ---------
def quick_validate_item(item: Dict) -> Tuple[bool, List[str]]:
    """
    Лёгкая проверка валидности для генерации (не заменяет твой большой валидатор):
    - q есть, v размер 2..4, ответы ≤2 токенов;
    - n есть и совпадает (норм.) с одним из v[*].a;
    - запрет утечки: ответ не подстрока контекста (lower());
    - ответы различны (норм.).
    """
    errs: List[str] = []
    q = str(item.get("q", "")).strip()
    v = item.get("v") or []
    n = item.get("n", None)
    if not q:
        errs.append("q.missing")
    if not isinstance(v, list) or not (2 <= len(v) <= 4):
        errs.append("v.size")
    answers_norm = []
    if isinstance(v, list):
        for i, vv in enumerate(v):
            if not isinstance(vv, dict):
                errs.append(f"v[{i}].type")
                continue
            c = (vv.get("c") or "").strip()
            a = (vv.get("a") or "").strip()
            if not c:
                errs.append(f"v[{i}].c.empty")
            if not a:
                errs.append(f"v[{i}].a.empty")
            if len(tokenize_words(a)) > 10:
                errs.append(f"v[{i}].a.too_long")
            # утечка
            if a and c and a.lower() in c.lower():
                errs.append(f"v[{i}].leak")
            # нормализация ответа (простая)
            a_norm = re.sub(r"^(?:the|a|an)\s+", "", a.strip(), flags=re.I).lower()
            answers_norm.append(a_norm)
    # различимость ответов
    if answers_norm and len(set(answers_norm)) != len(answers_norm):
        errs.append("v.answers.dup")
    # базовый ответ
    if n is None or str(n).strip() == "":
        errs.append("n.missing")
    else:
        n_norm = re.sub(r"^(?:the|a|an)\s+", "", str(n).strip(), flags=re.I).lower()
        if answers_norm and n_norm not in answers_norm:
            errs.append("n.not_aligned")

    return (len(errs) == 0), errs


def partition_and_annotate(items: List[Dict], field: str, subfield: str, subject: str, lang: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Добавляет служебные поля и теги аномалий:
      - для всех: f/s/j/d;
      - для всех: t = список кодов аномалий quick_validate_item (пустой список для валидных).
    Возвращает (valid_items, invalid_items).
    """
    valid: List[Dict] = []
    invalid: List[Dict] = []
    ts = now_iso()
    for it in items:
        ok, errs = quick_validate_item(it)
        it["f"] = field
        it["s"] = subfield
        it["j"] = subject
        it["d"] = ts
        it["t"] = list(errs) if not ok else []
        # язык можно добавить при необходимости: it["lang"] = lang
        if ok:
            valid.append(it)
        else:
            invalid.append(it)
    return valid, invalid


# --------- Промпт ---------
def build_generation_prompt(field: str, subfield: str, subject: str, num_q: int) -> str:
    return f"""
System
You are a careful generator of context-dependent QA items.
Return ONLY a JSON object with one key "items", where "items" is an array of {num_q} objects. Each object represents one question with multiple paradigm-dependent interpretations. No extra text/markdown.

User
Objective: In the subject "{subject}" of the subfield "{subfield}" in "{field}", generate {num_q} items. Each item has one question whose answer depends on the underlying rules/assumptions (paradigm), not on missing facts.

Hard rules (must all hold):
R1. Each question has 2–4 contexts in "v". Contexts must change the governing interpretation (different rules/definitions/units/number systems/paradigms), not just add examples or facts. Everyday domains are allowed only if they imply a distinct rule-system.
R2. Baseline answer "n":
    R2a. "n" is the answer under the conventional (default) interpretation commonly assumed for the question.
    R2b. Exactly one context in "v" must instantiate that same default interpretation, so that "n" == its "a".
    R2c. All other contexts must be non-default interpretations yielding different answers.
R3. Answers are DISTINCT across contexts after simple normalization: lowercase, trim, remove articles ("a/an/the").
R4. No leakage: a context must NOT contain its own answer tokens (substring match after lowercasing).
R5. Orthogonality: contexts must be semantically different (domains/definitions/number systems/units/paradigms), not minor variants.
R6. Finite diversity: forbid vague answers like "depends", "unknown", "varies".
R7. Concision: each answer ≤ 2 tokens (words or numbers); use digits and canonical symbols where applicable.
R8. Time-neutrality: avoid time/popularity/superlatives unless an explicit year (YYYY) is present in the context.
R9. Epistemic scope: keep within the stated field/subfield/subject unless a context explicitly frames another formal paradigm.
R10. Self-containment: each context is a short phrase (≤ 12 words) that changes assumptions. Do not hint at the answer; do not quote it.

Self-check BEFORE output (do not print this checklist):
C1. "q" is clear and yields "n" under a default interpretation.
C2. Exactly one context matches the default so that "n" equals one "a".
C3. All "a" unique after normalization; each ≤ 2 tokens.
C4. No "a" appears as a substring in its own "c" (lowercased).
C5. Contexts change rules/units/definitions and are mutually non-overlapping.
C6. No time-sensitive phrasing without explicit year.

Output format (strict JSON):
{{
  "items": [
    {{
      "q": "Your question",
      "n": "Baseline answer",
      "v": [
        {{"c": "Context 1", "a": "Answer 1"}},
        {{"c": "Context 2", "a": "Answer 2"}}
      ]
    }}
  ]
}}

Example:
[{{
  "q": "What is the value of 2×2?",
  "n": "4",
  "v": [
    {{"c": "In standard decimal arithmetic", "a": "4"}},
    {{"c": "In arithmetic modulo 3", "a": "1"}},
    {{"c": "Shown in base-4 numerals", "a": "10"}}
  ]
}},
{{
  "q": "What color model encodes a single on-screen element?",
  "n": "RGB",
  "v": [
    {{"c": "In emissive display pipelines", "a": "RGB"}},
    {{"c": "In subtractive print workflows", "a": "CMYK"}},
    {{"c": "In cylindrical hue-saturation schemes", "a": "HSV"}}
  ]
}}]
Generate exactly {num_q} objects inside "items" using ONLY the keys "q","n","v","c","a".
Prefer 3 contexts when natural; 2 is acceptable if both are clearly distinct paradigms.
""".strip()


# --------- OpenAI клиент ---------
def get_client(api_key: str):
    # openai>=1.0 стиль
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def plan_batch_size(prompt_tokens: int, max_tokens: int, requested: int, est_per_item: int = 90) -> int:
    """
    Простейший планировщик размера батча.
    - est_per_item: грубая оценка токенов на 1 объект (ответы+контексты)
    - держим запас 256 токенов на «служебку»
    """
    budget = max(0, int(max_tokens) - 256)
    if budget <= 0:
        return 1
    cap = max(1, budget // max(1, est_per_item))
    return max(1, min(requested, cap))


# --- replace function signature & body of call_openai() ---
def call_openai(
    client,
    model: str,
    prompt: str,
    max_tokens: int = 5000,
    temperature: float = 0.7,
    seed: Optional[int] = None,
    log_prompt: bool = False,
    log_response: bool = False,
    truncate: int = 2000,
    expected_batch: Optional[int] = None,  # <— НОВОЕ
) -> Tuple[str, Dict[str, Any]]:

    # сообщения — явно типизируем
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": "You are an AI language model that generates questions."},
        {"role": "user", "content": prompt},
    ]

    # --- ВАЖНО: params как dict[str, Any], иначе типизатор «зажмёт» тип значений
    params: Dict[str, Any] = {"model": model, "messages": msgs}
    # просим сразу JSON-объект (уменьшает накладные расходы против json_schema)
    params["response_format"] = {"type": "json_schema", "json_schema": qa_batch_schema(expected_batch)}

    # температура: не отправляем 1.0, т.к. часть моделей поддерживает только дефолт
    if temperature is not None and float(temperature) != 1.0:
        params["temperature"] = float(temperature)

    # первая попытка — modern: max_completion_tokens
    params["max_completion_tokens"] = int(max_tokens)
    if seed is not None:
        params["seed"] = int(seed)

    def _try(p: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            resp, status, headers, raw_text = _create_with_raw_response(client, **p)
        except TypeError as te:
            # на всякий случай убираем response_format и пробуем снова
            if "response_format" in p:
                p = dict(p)
                p.pop("response_format", None)
                resp, status, headers, raw_text = _create_with_raw_response(client, **p)
            else:
                raise

        # Метаданные HTTP
        if status is not None:
            # печатаем только пару безопасных заголовков
            safe_headers = {k: v for k, v in (headers or {}).items() if k.lower() in {"content-type", "x-request-id"}}
            raw_head = f"status={status}\n" + "\n".join(f"{k}: {v}" for k, v in safe_headers.items())
            print_block("[RAW HTTP META]", raw_head, max_len=2000, color=typer.colors.YELLOW)
            if raw_text:
                # тело может быть большим — но это ровно то, что ты просил
                print_block("[RAW HTTP BODY]", raw_text, max_len=8000, color=typer.colors.BRIGHT_BLACK)

        _dump_choice_debug(resp)
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(resp, "usage", None)
        meta: Dict[str, Any] = {
            "id": getattr(resp, "id", None),
            "model": getattr(resp, "model", None),
            "finish_reason": finish,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
        # Явная диагностика пустого контента
        if not content.strip():
            # Печатаем короткий маркер, чтобы в логах было видно сразу
            print_block("[GPT RESPONSE EMPTY]", f"finish_reason={finish}", max_len=200, color=typer.colors.RED)
            raise ValueError(f"Empty content (finish_reason={finish})")
        if log_response:
            meta_str = f"(model={resp.model}, finish_reason={resp.choices[0].finish_reason})"
            print_block(f"[GPT RESPONSE {meta_str}]", content, max_len=truncate, color=typer.colors.GREEN)
        return content, meta

    # лог промпта
    if log_prompt:
        print_block("[PROMPT → GPT]", prompt, max_len=truncate, color=typer.colors.CYAN)

    try:
        return _try(params)
    except Exception as e1:
        # Если SDK пробросил HTTP-ошибку — попробуем вывести тело
        resp_obj = getattr(e1, "response", None)
        if resp_obj is not None:
            try:
                status = getattr(resp_obj, "status_code", None)
                text = getattr(resp_obj, "text", None)
                if text:
                    print_block("[RAW HTTP ERROR BODY]", f"status={status}\n{text}", max_len=8000, color=typer.colors.RED)
            except Exception:
                pass
        msg = f"{getattr(e1, 'message', '')} {e1}"

        # температура не поддерживается — убрать
        if "unsupported_value" in msg and "temperature" in msg:
            params.pop("temperature", None)
            try:
                return _try(params)
            except Exception as e1c:
                msg = f"{getattr(e1c, 'message', '')} {e1c}"

        # seed не поддерживается — убрать
        if "seed" in params:
            params.pop("seed", None)
            try:
                return _try(params)
            except Exception as e1b:
                msg = f"{getattr(e1b, 'message', '')} {e1b}"

        # переключение на legacy max_tokens
        if "unsupported_parameter" in msg or "max_completion_tokens" in msg:
            params.pop("max_completion_tokens", None)
            params["max_tokens"] = int(max_tokens)
            try:
                return _try(params)
            except Exception as e2:
                msg2 = f"{getattr(e2, 'message', '')} {e2}"
                # повторно убираем temperature/seed на всякий
                params.pop("temperature", None)
                params.pop("seed", None)
                return _try(params)
        raise


def backoff_sleep(attempt: int, base: float = 1.5, jitter: float = 0.25):
    # экспоненциальная пауза с джиттером
    t = (base**attempt) + random.uniform(0, jitter)
    time.sleep(min(t, 20.0))


# --------- Основная генерация ---------
def generate_for_subject(
    client,
    model: str,
    lang: str,
    field: str,
    subfield: str,
    subject: Dict,
    num_q: int,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 3000,
    seed: Optional[int] = None,
    log_prompt: bool = False,
    log_response: bool = False,
    log_json: bool = False,
    truncate: int = 2000,
) -> Tuple[List[Dict], List[Dict]]:
    subj_name = subject.get("subject", "")
    cur_num_q = int(num_q)
    remaining = int(num_q)
    typer.secho(f"\n[GEN] {field} / {subfield} / {subj_name} -> {cur_num_q}", fg=typer.colors.CYAN)

    results: List[Dict] = []
    results_valid: List[Dict] = []
    results_invalid: List[Dict] = []
    last_err = None
    attempt = 0
    while remaining > 0 and attempt <= max_retries:
        try:
            # планируем безопасный размер батча
            # примечание: prompt токены не знаем заранее → берём эвристику
            batch = plan_batch_size(prompt_tokens=600, max_tokens=max_tokens, requested=remaining, est_per_item=90)
            typer.secho(f"[BATCH] target={remaining}, batch={batch}", fg=typer.colors.MAGENTA)

            prompt = build_generation_prompt(field, subfield, subj_name, batch)
            raw, meta = call_openai(
                client,
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                log_prompt=log_prompt,
                log_response=log_response,
                truncate=truncate,
            )
            # показать извлечённый JSON-блок
            extracted = extract_json_block(raw)
            if log_json:
                print_block("[EXTRACTED JSON BLOCK]", extracted, max_len=truncate, color=typer.colors.MAGENTA)

            items = parse_json_items(extracted)
            if not items:
                raise ValueError(
                    f"Empty generation result: parsed=0 (raw_len={len(raw)}, extracted_len={len(extracted)}; finish_reason={meta.get('finish_reason')}; completion_tokens={meta.get('completion_tokens')})"
                )

            valid, invalid = partition_and_annotate(items, field, subfield, subj_name, lang)
            # Логируем кратко статистику батча
            if invalid:
                typer.secho(f"[BATCH] valid={len(valid)} invalid={len(invalid)}", fg=typer.colors.YELLOW)
            else:
                typer.secho(f"[BATCH] valid={len(valid)}", fg=typer.colors.GREEN)

            if not valid and not invalid:
                raise ValueError("Empty generation result after partition.")

            results_valid.extend(valid)
            results_invalid.extend(invalid)
            remaining -= len(valid)  # считаем план по валидным
            attempt = 0  # успешный батч сбрасывает счётчик ошибок
            continue
        except Exception as e:
            last_err = e
            typer.secho(f"[WARN] Attempt {attempt+1}/{max_retries+1} failed: {e}", fg=typer.colors.YELLOW)
            attempt += 1
            msg = str(e)
            # эвристика: либо finish_reason=length, либо completion_tokens≈max_tokens
            if ("finish_reason=length" in msg) or ("Empty content" in msg) or ("parsed=0" in msg):
                # ужимаем оценку per-item, чтобы следующий plan_batch_size дал меньший батч
                # (простой способ — снизить max_tokens локально, если можно)
                if max_tokens > 1024:
                    max_tokens = max(1024, int(max_tokens * 0.9))
                    typer.secho(f"[ADAPT] Shrink max_tokens to {max_tokens}", fg=typer.colors.MAGENTA)

            if attempt < max_retries:
                backoff_sleep(attempt + 1)
            else:
                break
    if results_valid or results_invalid:
        return results_valid, results_invalid
    raise RuntimeError(f"Generation failed: {last_err}")


# --------- Typer CLI ---------
@app.command("gen")
def cmd_gen(
    field_data_file: str = typer.Option("field_data.json", help="Имя файла с FoK-данными (в каталоге DATA_DIR)."),
    out: str = typer.Option(None, help="Путь вывода JSONL. По умолчанию DATA_DIR/{model}_generated_questions_{lang}.jsonl"),
    model: str = typer.Option("gpt-4o", help=f"Имя модели. Поддерживаемые: {', '.join(SUPPORTED_MODELS)}"),
    lang: str = typer.Option("en", help="Код языка для пометки данных в имени файла."),
    num_questions: int = typer.Option(50, min=1, max=50, help="Сколько вопросов на один subject."),
    temperature: float = typer.Option(1.0, min=0.0, max=2.0, help="Температура генерации. Некоторые модели поддерживают только значение 1; в таком случае параметр будет опущен."),
    max_tokens: int = typer.Option(7000, min=512, max=8192),
    seed: Optional[int] = typer.Option(None, help="Фиксировать seed (если поддерживается)."),
    max_retries: int = typer.Option(3, min=0, max=10),
    resume: bool = typer.Option(True, help="Пропускать уже обработанные (по f/s/j) в выходном файле."),
    max_subfields: int = typer.Option(2, min=0, help="Максимум subfields за один запуск (0 = без ограничений)."),
    log_prompt: bool = typer.Option(True, help="Печатать в терминал отправленный промпт."),
    log_response: bool = typer.Option(True, help="Печатать сырые ответы модели."),
    log_json: bool = typer.Option(True, help="Печатать извлечённый JSON-блок."),
    truncate: int = typer.Option(2000, min=200, max=20000, help="Макс. длина печатаемых блоков."),
):
    """
    Генерирует вопросы по всем subjects из field_data.json и дописывает в JSONL.
    """
    api_key = read_api_key()
    if not api_key:
        typer.secho("Error: OPENAI_API_KEY not found in .secrets.toml", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # пути
    field_data_path = os.path.join(DATA_DIR, field_data_file)
    if out is None:
        out = os.path.join(DATA_DIR, DEFAULT_OUT_TMPL.format(model=model, lang=lang))

    # загрузка
    field_data_list = load_field_data(field_data_path)
    processed = load_processed_subjects(out) if resume else set()

    client = get_client(api_key)

    # обход FoK → sfok → subjects
    total = 0
    skipped = 0
    subfields_processed = 0
    stop_all = False
    for fok in field_data_list:
        field = fok.get("fok", "Unknown Field")
        sfoks = fok.get("sfoks", [])
        for sfok in sfoks:
            if stop_all:
                break

            subfield = sfok.get("name", "Unknown Subfield")
            subjects = sfok.get("subjects", [])

            # собрать список к обработке с учётом resume
            subjects_to_process = []
            for subject in subjects:
                key = (field.strip(), subfield.strip(), str(subject.get("subject", "")).strip())
                if resume and key in processed:
                    skipped += 1
                    typer.secho(f"[SKIP] already processed: {key}", fg=typer.colors.BLUE)
                    continue
                subjects_to_process.append(subject)

            # если в этом subfield нечего делать — не засчитываем в квоту
            if not subjects_to_process:
                continue

            # проверить лимит subfields
            if max_subfields and subfields_processed >= max_subfields:
                typer.secho(f"[INFO] Reached max_subfields={max_subfields}. Stopping.", fg=typer.colors.MAGENTA)
                stop_all = True
                break

            # обработать все subjects этого subfield
            for subject in subjects_to_process:
                key = (field.strip(), subfield.strip(), str(subject.get("subject", "")).strip())
                try:
                    valid_items, invalid_items = generate_for_subject(
                        client=client,
                        model=model,
                        lang=lang,
                        field=field,
                        subfield=subfield,
                        subject=subject,
                        num_q=num_questions,
                        max_retries=max_retries,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed,
                        log_prompt=log_prompt,
                        log_response=log_response,
                        log_json=log_json,
                        truncate=truncate,
                    )
                    # пишем валидные
                    if valid_items:
                        write_jsonl(out, valid_items)
                        total += len(valid_items)
                        typer.secho(f"[OK] wrote {len(valid_items)} items → {out}", fg=typer.colors.GREEN)
                    # пишем невалидные в *_inval.jsonl
                    if invalid_items:
                        inval_out = out[:-6] + "_inval.jsonl" if out.endswith(".jsonl") else out + "_inval.jsonl"
                        write_jsonl(inval_out, invalid_items)
                        typer.secho(f"[OK] wrote {len(invalid_items)} invalid items → {inval_out}", fg=typer.colors.MAGENTA)
                except Exception as e:
                    typer.secho(f"[ERR] {key}: {e}", fg=typer.colors.RED)

            # subfield полностью обработан — засчитываем в квоту
            subfields_processed += 1

    typer.secho(f"\nDone. new items: {total}, skipped subjects: {skipped}", fg=typer.colors.GREEN)


@app.command("resume")
def cmd_resume_list(
    out: str = typer.Option(..., help="Путь к уже заполненному JSONL"),
    limit: int = typer.Option(20, help="Сколько ключей показать"),
):
    """Печатает (f,s,j) ключи уже обработанных subjects в выходном JSONL."""
    done = load_processed_subjects(out)
    typer.echo(f"Processed subjects: {len(done)}")
    for i, key in enumerate(sorted(done)):
        if i >= limit:
            typer.echo("...")
            break
        typer.echo(f"- {key}")


@app.command("dryrun")
def cmd_dryrun(
    field: str = typer.Option("Natural Sciences", help="Field of knowledge"),
    subfield: str = typer.Option("Mathematics", help="Subfield"),
    subject: str = typer.Option("Infinity", help="Subject"),
    num_questions: int = typer.Option(3, min=1, max=10),
    model: str = typer.Option("gpt-4o"),
    temperature: float = typer.Option(0.7),
    max_tokens: int = typer.Option(2000),
    seed: Optional[int] = typer.Option(None),
    log_prompt: bool = typer.Option(True, help="Печатать промпт."),
    log_response: bool = typer.Option(True, help="Печатать ответы."),
    log_json: bool = typer.Option(True, help="Печатать извлечённый JSON-блок."),
    truncate: int = typer.Option(2000, min=200, max=20000),
):
    """Пробная генерация по одному subject без записи в файл (печатает результат в stdout)."""
    api_key = read_api_key()
    if not api_key:
        typer.secho("Error: OPENAI_API_KEY not found in .secrets.toml", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    client = get_client(api_key)
    subj = {"subject": subject}
    valid_items, invalid_items = generate_for_subject(
        client,
        model,
        "en",
        field,
        subfield,
        subj,
        num_q=num_questions,
        max_retries=2,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        log_prompt=log_prompt,
        log_response=log_response,
        log_json=log_json,
        truncate=truncate,
    )
    typer.secho("\n[VALID ITEMS]", fg=typer.colors.GREEN)
    typer.echo(json.dumps(valid_items, ensure_ascii=False, indent=2))
    if invalid_items:
        typer.secho("\n[INVALID ITEMS]", fg=typer.colors.MAGENTA)
        typer.echo(json.dumps(invalid_items, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
