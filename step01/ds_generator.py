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
from typing import Dict, List, Tuple, Optional

import typer
import toml

# --------- Константы / пути ---------
app = typer.Typer(add_completion=False, no_args_is_help=True)
DATA_DIR = "./step01/data"
SECRETS = ".secrets.toml"
DEFAULT_OUT_TMPL = "{model}_generated_questions_{lang}.jsonl"
SUPPORTED_MODELS = [
    "gpt-5",         # placeholder для будущих настроек
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
]

# --------- Утилиты ---------
WORD_RE = re.compile(r"\w+", flags=re.U | re.M)

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
    """Вырезает JSON-массив из ответа: убирает ```...``` и текст вокруг."""
    s = (reply or "").strip()
    if s.startswith("```json"):
        s = s[7:]
        if s.endswith("```"):
            s = s[:-3]
    elif s.startswith("```"):
        s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if m:
        return m.group(0).strip()
    # попытка спасти одиночный объект
    m2 = re.search(r"\{.*\}", s, re.DOTALL)
    if m2:
        return f"[{m2.group(0).strip()}]"
    return s

def parse_json_items(reply: str) -> List[Dict]:
    s = extract_json_block(reply)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return []
    except json.JSONDecodeError:
        # fallback: выдёргиваем объекты
        parts = re.findall(r"\{.*?\}(?=,\s*\{|\s*$)", s, re.DOTALL)
        out: List[Dict] = []
        for i, p in enumerate(parts):
            try:
                out.append(json.loads(p))
            except json.JSONDecodeError:
                typer.secho(f"[WARN] Skip broken item #{i}", fg=typer.colors.YELLOW)
        return out

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
            key = (str(obj.get("f","")).strip(),
                   str(obj.get("s","")).strip(),
                   str(obj.get("j","")).strip())
            if all(key):
                done.add(key)
    return done

def write_jsonl(path: str, items: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

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
    q = str(item.get("q","")).strip()
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
            if len(tokenize_words(a)) > 2:
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

def postprocess_and_filter(items: List[Dict],
                           field: str, subfield: str, subject: str,
                           lang: str) -> List[Dict]:
    """Добавляет f/s/j/d и отбрасывает грубо невалидные записи."""
    out: List[Dict] = []
    ts = now_iso()
    for it in items:
        ok, errs = quick_validate_item(it)
        if not ok:
            typer.secho(f"[WARN] Drop invalid item: {errs}", fg=typer.colors.YELLOW)
            continue
        it["f"] = field
        it["s"] = subfield
        it["j"] = subject
        it["d"] = ts
        # язык сейчас не добавляю, но можно it["lang"]=lang
        out.append(it)
    return out

# --------- Промпт ---------
def build_generation_prompt(field: str, subfield: str, subject: str, num_q: int) -> str:
    return f"""
System
You are a careful generator of context-dependent QA items. Output ONLY valid JSON as specified. Do not add explanations, comments, markdown, or extra keys.

User
Objective: In the subject "{subject}" of the subfield "{subfield}" in "{field}", generate {num_q} questions where the answer depends on the context or set of assumptions.

Hard rules (must all hold):
R1. Each question has 2–4 contexts in "v". Contexts must change the interpretation (paradigm/assumptions), not merely add missing facts.
R2. Provide a baseline answer "n" (answer with no context). It MUST be semantically the same as one of the context answers in "v".
R3. Answers are DISTINCT across contexts after simple normalization: lowercase, trim, remove articles ("a/an/the").
R4. No leakage: a context must NOT contain its answer tokens (substring match after lowercasing).
R5. Orthogonality: contexts must be semantically different (domains/definitions/number systems/units/paradigms).
R6. Finite diversity: forbid vague answers like "depends", "unknown", "varies".
R7. Concision: each answer ≤ 2 tokens (words or numbers); use digits and canonical symbols where applicable.
R8. Time-neutrality: avoid time/popularity/superlatives unless an explicit year (YYYY) is present in the context.
R9. Epistemic scope: keep within the stated field/subfield/subject unless the context explicitly frames another paradigm.
R10. Self-containment: each context is a short phrase (≤ 12 words) that changes assumptions. No quotes of the answer.

Self-check BEFORE output (do not print this checklist):
C1. "q" is clear and yields "n" without contexts. C2. "n" equals exactly one "a" in "v" after normalization.
C3. All "a" unique (normalized), each ≤2 tokens. C4. No "a" substring inside its paired "c" (lowercased).
C5. Contexts are mutually non-overlapping in meaning. C6. No time-sensitive phrasing without explicit year.

Output format (strict JSON):
[
  {{
    "q": "Your question",
    "n": "Baseline answer",
    "v": [
      {{"c": "Context 1", "a": "Answer 1"}},
      {{"c": "Context 2", "a": "Answer 2"}}
    ]
  }}
]

Generate exactly {num_q} objects in a JSON array using ONLY the keys "q", "n", "v", "c", "a".
""".strip()

# --------- OpenAI клиент ---------
def get_client(api_key: str):
    # openai>=1.0 стиль
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def call_openai(client, model: str, prompt: str,
                max_tokens: int = 3000, temperature: float = 0.7,
                seed: Optional[int] = None) -> str:
    msgs = [
        {"role": "system", "content": "You are an AI language model that generates questions."},
        {"role": "user", "content": prompt},
    ]
    params = dict(model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature)
    # необязательный seed (не все модели поддержат; игнор безопасен)
    if seed is not None:
        params["seed"] = seed  # type: ignore
    resp = client.chat.completions.create(**params)
    return resp.choices[0].message.content or ""

def backoff_sleep(attempt: int, base: float = 1.5, jitter: float = 0.25):
    # экспоненциальная пауза с джиттером
    t = (base ** attempt) + random.uniform(0, jitter)
    time.sleep(min(t, 20.0))

# --------- Основная генерация ---------
def generate_for_subject(client,
                         model: str,
                         lang: str,
                         field: str,
                         subfield: str,
                         subject: Dict,
                         num_q: int,
                         max_retries: int = 3,
                         temperature: float = 0.7,
                         max_tokens: int = 3000,
                         seed: Optional[int] = None) -> List[Dict]:
    subj_name = subject.get("subject", "")
    prompt = build_generation_prompt(field, subfield, subj_name, num_q)
    typer.secho(f"\n[GEN] {field} / {subfield} / {subj_name} -> {num_q}", fg=typer.colors.CYAN)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            raw = call_openai(client, model, prompt, max_tokens=max_tokens,
                              temperature=temperature, seed=seed)
            items = parse_json_items(raw)
            if not items:
                raise ValueError("Empty generation result.")
            items_pp = postprocess_and_filter(items, field, subfield, subj_name, lang)
            if not items_pp:
                raise ValueError("All generated items failed quick validation.")
            return items_pp
        except Exception as e:
            last_err = e
            typer.secho(f"[WARN] Attempt {attempt+1}/{max_retries+1} failed: {e}", fg=typer.colors.YELLOW)
            if attempt < max_retries:
                backoff_sleep(attempt+1)
            else:
                break
    raise RuntimeError(f"Generation failed after {max_retries+1} attempts: {last_err}")

# --------- Typer CLI ---------
@app.command("gen")
def cmd_gen(
    field_data_file: str = typer.Option("field_data.json", help="Имя файла с FoK-данными (в каталоге DATA_DIR)."),
    out: str = typer.Option(None, help="Путь вывода JSONL. По умолчанию DATA_DIR/{model}_generated_questions_{lang}.jsonl"),
    model: str = typer.Option("gpt-5", help=f"Имя модели. Поддерживаемые: {', '.join(SUPPORTED_MODELS)}"),
    lang: str = typer.Option("en", help="Код языка для пометки данных в имени файла."),
    num_questions: int = typer.Option(30, min=1, max=50, help="Сколько вопросов на один subject."),
    temperature: float = typer.Option(0.7, min=0.0, max=2.0),
    max_tokens: int = typer.Option(3000, min=512, max=8192),
    seed: Optional[int] = typer.Option(None, help="Фиксировать seed (если поддерживается)."),
    max_retries: int = typer.Option(3, min=0, max=10),
    resume: bool = typer.Option(True, help="Пропускать уже обработанные (по f/s/j) в выходном файле."),
    max_subfields: int = typer.Option(2, min=0, help="Максимум subfields за один запуск (0 = без ограничений)."),
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
                key = (field.strip(), subfield.strip(), str(subject.get("subject","")).strip())
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
                key = (field.strip(), subfield.strip(), str(subject.get("subject","")).strip())
                try:
                    items = generate_for_subject(
                        client=client, model=model, lang=lang,
                        field=field, subfield=subfield, subject=subject,
                        num_q=num_questions, max_retries=max_retries,
                        temperature=temperature, max_tokens=max_tokens, seed=seed
                    )
                    write_jsonl(out, items)
                    total += len(items)
                    typer.secho(f"[OK] wrote {len(items)} items → {out}", fg=typer.colors.GREEN)
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
):
    """Пробная генерация по одному subject без записи в файл (печатает результат в stdout)."""
    api_key = read_api_key()
    if not api_key:
        typer.secho("Error: OPENAI_API_KEY not found in .secrets.toml", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    client = get_client(api_key)
    subj = {"subject": subject}
    items = generate_for_subject(client, model, "en", field, subfield, subj,
                                 num_q=num_questions, max_retries=2,
                                 temperature=temperature, max_tokens=max_tokens, seed=seed)
    typer.echo(json.dumps(items, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    app()
