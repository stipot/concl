# validate_dataset.py
# Общий каркас валидации проектных записей датасета (контекстно-зависимые вопросы).
# Формат входных записей (JSONL построчно), минимально:
# {
#   "f": "<field>", "s": "<subfield>", "j": "<subject>",
#   "q": "<question>",
#   "v": [{"c":"<context1>", "a":"<answer1>"}, {"c":"<context2>", "a":"<answer2>"}],
#   "n": "<answer_without_context>"  # опционально
# }
#
# Выход:
# - JSONL с результатами по каждой записи (+ рекомендации по исправлению)
# - Сводный JSON с метриками покрытия/качества
#
""" python validate_dataset.py \
  --input ./step01/data/questions_data_en.jsonl \
  --out ./step01/data/validation_results.jsonl \
  --summary ./step01/data/validation_summary.json \
  --use-embeddings  """
# Опционально: проверка ортогональности контекстов через эмбеддинги (OpenAI).
# Для этого положите ключ в .secrets.toml под ключом OPENAI_API_KEY.

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import math
import os
import re
import statistics
import sys
import textwrap
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import toml  # pip install toml
except Exception:
    toml = None

# =========================
# Утилиты
# =========================

_WORD_RE = re.compile(r"\w+", flags=re.U | re.M)

def read_api_key_from_secrets(path: str = ".secrets.toml") -> Optional[str]:
    if toml is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        secrets = toml.load(path)
        return secrets.get("OPENAI_API_KEY")
    except Exception:
        return None

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_words(s: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(s)]

def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)

def cosine(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    nu = math.sqrt(sum(x*x for x in u))
    nv = math.sqrt(sum(x*x for x in v))
    if nu == 0 or nv == 0:
        return 0.0
    dot = sum(x*y for x, y in zip(u, v))
    return dot / (nu * nv)

# =========================
# Модель данных
# =========================

@dataclasses.dataclass
class Variation:
    c: str  # context
    a: str  # answer

@dataclasses.dataclass
class QAItem:
    f: Optional[str]
    s: Optional[str]
    j: Optional[str]
    q: str
    v: List[Variation]
    n: Optional[str] = None
    meta: Dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_raw(raw: Dict) -> "QAItem":
        if not isinstance(raw, dict):
            raise ValueError("raw item is not a JSON object (dict)")

        # Запись должна содержать минимум q и v (список пар {c,a})
        q = str(raw.get("q", "")).strip()
        v_raw = raw.get("v", [])
        if not isinstance(v_raw, list):
            raise ValueError("field 'v' must be a list")

        v_list: List[Variation] = []
        for vv in v_raw:
            if isinstance(vv, dict):
                c = str(vv.get("c", "")).strip()
                a = str(vv.get("a", "")).strip()
                v_list.append(Variation(c=c, a=a))

        return QAItem(
            f=raw.get("f"),
            s=raw.get("s"),
            j=raw.get("j"),
            q=q,
            v=v_list,
            n=(str(raw.get("n")).strip() if raw.get("n") is not None else None),
            meta={k: v for k, v in raw.items() if k not in {"f", "s", "j", "q", "v", "n"}}
        )

def load_jsonl(path: str) -> List[Dict]:
    """
    Гибкая загрузка:
    - JSONL: по строке -> dict/список -> добавляем объекты-словарики
    - если ничего не прочли — пробуем считать весь файл как единый JSON
      (массив объектов или один объект)
    - пропускаем строки-строки, обрывки массивов, сводки и т.п.
    """
    data: List[Dict] = []
    bad_lines = 0

    def _extend_from_obj(obj):
        nonlocal data
        if isinstance(obj, dict):
            # берем только «проектные» записи, где есть поле 'q' и 'v'
            if "q" in obj and "v" in obj:
                data.append(obj)
        elif isinstance(obj, list):
            for x in obj:
                if isinstance(x, dict) and "q" in x and "v" in x:
                    data.append(x)
        # игнорируем прочие типы (str, int, и т.д.)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Нередко попадаются обрывки массивов или сводок; отсечем быстрыми эвристиками
            if s in {"]", "],", "[", "[,", "}", "},"}:
                bad_lines += 1
                print(f"[WARN] Bad JSONL line skipped: {s[:120]}...", file=sys.stderr)
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_lines += 1
                print(f"[WARN] Bad JSONL line skipped: {s[:120]}...", file=sys.stderr)
                continue
            _extend_from_obj(obj)

    # Если после построчного парсинга ничего не прочли — пробуем целиком
    if not data:
        with open(path, "r", encoding="utf-8") as f:
            whole = f.read().strip()
        try:
            obj = json.loads(whole)
            _extend_from_obj(obj)
        except Exception:
            pass  # оставим пустым

    if not data:
        print("[WARN] No valid project records found in input file.", file=sys.stderr)
    else:
        print(f"[INFO] Loaded {len(data)} valid records"
              + (f", skipped {bad_lines} noisy lines." if bad_lines else "."))

    return data


# =========================
# Эмбеддинги (опционально)
# =========================

class EmbeddingProvider:
    """Абстракция эмбеддингов. Реализаций может быть несколько (OpenAI, локальные и т.д.)."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: Optional[str], model: str = "text-embedding-3-small"):
        if api_key is None:
            raise RuntimeError("OpenAI API key is required for OpenAIEmbeddingProvider.")
        try:
            import openai  # openai>=1.0
        except Exception as e:
            raise RuntimeError("Please install openai>=1.0 to use OpenAI embeddings.") from e
        # new client style
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        # API accepts up to a certain batch size; keep it simple here.
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

class DummyEmbeddingProvider(EmbeddingProvider):
    """Заглушка: возвращает нули (ортогональность работать не будет, но валидатор можно отключить)."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 10 for _ in texts]

# =========================
# Результаты валидации
# =========================

@dataclasses.dataclass
class ValidationIssue:
    code: str
    title: str
    severity: str  # "error" | "warn" | "info"
    detail: Optional[str] = None
    suggest_fix: Optional[str] = None
    meta: Dict = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class ValidationResult:
    item: QAItem
    passed: bool
    issues: List[ValidationIssue]
    metrics: Dict[str, float]  # любые количественные метрики
    recommendations: List[str]

# =========================
# Базовый валидатор и конкретные проверки
# =========================

class BaseValidator:
    name: str = "base"

    def validate(self, item: QAItem) -> List[ValidationIssue]:
        raise NotImplementedError

class StructureValidator(BaseValidator):
    """Структурная полнота: вопрос, >=2 контекстов, ответы не пустые, короткие ответы и т.п."""
    name = "structure"

    def __init__(self, min_contexts: int = 2, max_answer_tokens: int = 2):
        self.min_contexts = min_contexts
        self.max_answer_tokens = max_answer_tokens

    def validate(self, item: QAItem) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not item.q:
            issues.append(ValidationIssue("q.missing", "Нет вопроса", "error"))

        if len(item.v) < self.min_contexts:
            issues.append(ValidationIssue(
                "contexts.too_few",
                f"Мало контекстов: {len(item.v)} < {self.min_contexts}",
                "error",
                suggest_fix="Добавить недостающие контексты с альтернативными интерпретациями."
            ))

        for i, var in enumerate(item.v):
            if not var.c:
                issues.append(ValidationIssue(
                    "context.empty",
                    f"Пустой контекст v[{i}]",
                    "error",
                    suggest_fix="Сформулировать контекст так, чтобы он менял понимание вопроса."
                ))
            if not var.a:
                issues.append(ValidationIssue(
                    "answer.empty",
                    f"Пустой ответ v[{i}].a",
                    "error",
                    suggest_fix="Задать чёткий краткий ответ; не более 1–2 слов/чисел."
                ))
            # краткость ответа
            if var.a and len(tokenize_words(var.a)) > self.max_answer_tokens:
                issues.append(ValidationIssue(
                    "answer.too_long",
                    f"Слишком длинный ответ v[{i}]: «{var.a}»",
                    "warn",
                    suggest_fix="Сократить до ≤ 2 слов/чисел."
                ))

        # базовый ответ без контекста — необязателен, но желателен
        if item.n is None:
            issues.append(ValidationIssue(
                "baseline.missing",
                "Нет базового ответа (без контекста).",
                "warn",
                suggest_fix="Добавить поле n как якорный ответ без контекста."
            ))
        else:
            if len(tokenize_words(item.n)) > self.max_answer_tokens:
                issues.append(ValidationIssue(
                    "baseline.too_long",
                    f"Слишком длинный базовый ответ: «{item.n}»",
                    "warn",
                    suggest_fix="Сократить до ≤ 2 слов/чисел."
                ))

        return issues

class LeakValidator(BaseValidator):
    """Утечка ответа в контексте (контекст не должен явно содержать ответ)."""
    name = "leak"

    def __init__(self, case_insensitive: bool = True):
        self.case_insensitive = case_insensitive

    def validate(self, item: QAItem) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for i, var in enumerate(item.v):
            ans = normalize_text(var.a)
            ctx = var.c
            if not ans or not ctx:
                continue
            hay = ctx.lower() if self.case_insensitive else ctx
            needle = ans.lower() if self.case_insensitive else ans
            # простая эвристика: прямое вхождение ответа в контекст
            if needle and needle in hay:
                issues.append(ValidationIssue(
                    "leak.answer_in_context",
                    f"Ответ v[{i}].a явно присутствует в контексте.",
                    "warn",
                    detail=f'answer="{var.a}", context="{var.c}"',
                    suggest_fix="Переформулировать контекст так, чтобы он не подсказывал ответ буквально."
                ))
        return issues

class UniquenessValidator(BaseValidator):
    """Ответ одного контекста не должен подходить к другому (грубая проверка эквивалентности)."""
    name = "uniqueness"

    def validate(self, item: QAItem) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        answers = [normalize_text(v.a).lower() for v in item.v if v.a]
        # Дубли ответов у разных контекстов — риск неортогональности
        dup_counts = collections.Counter(answers)
        collisions = [ans for ans, cnt in dup_counts.items() if cnt > 1 and ans]
        if collisions:
            issues.append(ValidationIssue(
                "answers.duplicate",
                f"Одинаковые ответы для разных контекстов: {', '.join(collisions)}",
                "warn",
                suggest_fix="Переформулировать/расщепить контексты, чтобы ответы различались."
            ))
        return issues

class OrthogonalityValidator(BaseValidator):
    """Семантическая ортогональность контекстов по эмбеддингам (низкая косинусная близость)."""
    name = "orthogonality"

    def __init__(self, embedder: Optional[EmbeddingProvider], max_sim: float = 0.80):
        self.embedder = embedder
        self.max_sim = max_sim

    def validate(self, item: QAItem) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if self.embedder is None or len(item.v) < 2:
            return issues
        contexts = [normalize_text(v.c) for v in item.v if v.c]
        if len(contexts) < 2:
            return issues
        vecs = self.embedder.embed(contexts)
        n = len(vecs)
        too_close_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                sim = cosine(vecs[i], vecs[j])
                if sim >= self.max_sim:
                    too_close_pairs.append((i, j, sim))
        if too_close_pairs:
            pairs_str = ", ".join([f"({i},{j}): {sim:.2f}" for i, j, sim in too_close_pairs])
            issues.append(ValidationIssue(
                "contexts.too_similar",
                "Контексты семантически слишком близки (низкая ортогональность).",
                "warn",
                detail=pairs_str,
                suggest_fix="Объединить близкие контексты или развести их по смыслу/парадигме."
            ))
        return issues

class BaselineAlignmentValidator(BaseValidator):
    """Базовый ответ должен быть близок хотя бы к одному контекстному ответу (простая проверка равенства/нормализации)."""
    name = "baseline"

    def validate(self, item: QAItem) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if item.n is None or not item.v:
            return issues
        nrm_n = normalize_text(item.n).lower()
        answers = [normalize_text(v.a).lower() for v in item.v if v.a]
        if not answers:
            return issues
        if nrm_n and nrm_n not in answers:
            issues.append(ValidationIssue(
                "baseline.not_aligned",
                "Базовый ответ не согласован ни с одним контекстным ответом.",
                "warn",
                suggest_fix="Либо обновить базовый ответ, либо проверить формулировки контекстов/ответов."
            ))
        return issues

class FiniteDiversityValidator(BaseValidator):
    """Грубая проверка «конечности разнообразия»: слишком много контекстов/ответы-«ловушки»."""
    name = "finite"

    def __init__(self, max_contexts: int = 8):
        self.max_contexts = max_contexts

    def validate(self, item: QAItem) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if len(item.v) > self.max_contexts:
            issues.append(ValidationIssue(
                "contexts.too_many",
                f"Слишком много контекстов: {len(item.v)} > {self.max_contexts}",
                "warn",
                suggest_fix="Свести множество к конечному набору ортогональных контекстов."
            ))
        # эвристика: ответы вроде "depends", "varies", "unknown" ослабляют конечность
        vague = {"depends", "varies", "unknown", "unclear", "it depends"}
        for i, var in enumerate(item.v):
            if normalize_text(var.a).lower() in vague:
                issues.append(ValidationIssue(
                    "answer.vague",
                    f"Ответ v[{i}] неконкретный («{var.a}»).",
                    "warn",
                    suggest_fix="Заменить на конкретный конечный вариант."
                ))
        return issues

# =========================
# Оркестратор валидации
# =========================

class ValidatorPipeline:
    def __init__(self, validators: List[BaseValidator]):
        self.validators = validators

    def validate_item(self, item: QAItem) -> ValidationResult:
        issues: List[ValidationIssue] = []
        for v in self.validators:
            try:
                issues.extend(v.validate(item))
            except Exception as e:
                issues.append(ValidationIssue(
                    code=f"{v.name}.error",
                    title=f"Ошибка валидатора {v.name}",
                    severity="error",
                    detail=str(e)
                ))
        # Правило итогового статуса: нет error => passed=True
        passed = not any(iss.severity == "error" for iss in issues)

        # Простейшие метрики
        metrics = {
            "contexts": float(len(item.v)),
            "answers_unique": float(len(set(normalize_text(v.a).lower() for v in item.v if v.a))),
        }

        recommendations = []
        for iss in issues:
            if iss.suggest_fix and iss.suggest_fix not in recommendations:
                recommendations.append(iss.suggest_fix)

        return ValidationResult(item=item, passed=passed, issues=issues, metrics=metrics, recommendations=recommendations)

# =========================
# IO
# =========================

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                # игнорируем битые строки, но логируем в stderr
                print(f"[WARN] Bad JSONL line skipped: {line[:120]}...", file=sys.stderr)
    return data

def write_jsonl(path: str, items: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(jdump(it) + "\n")

def write_summary(path: str, summary: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(jdump(summary))

# =========================
# Главная процедура
# =========================

def build_pipeline(use_embeddings: bool, max_sim: float, max_contexts: int, max_answer_tokens: int) -> ValidatorPipeline:
    embedder: Optional[EmbeddingProvider] = None
    if use_embeddings:
        api_key = read_api_key_from_secrets()
        if api_key:
            try:
                embedder = OpenAIEmbeddingProvider(api_key=api_key)
            except Exception as e:
                print(f"[WARN] Embeddings disabled: {e}", file=sys.stderr)
                embedder = None
        else:
            print("[WARN] OPENAI_API_KEY not found; embeddings disabled.", file=sys.stderr)

    validators: List[BaseValidator] = [
        StructureValidator(min_contexts=2, max_answer_tokens=max_answer_tokens),
        LeakValidator(),
        UniquenessValidator(),
        OrthogonalityValidator(embedder=embedder, max_sim=max_sim),
        BaselineAlignmentValidator(),
        FiniteDiversityValidator(max_contexts=max_contexts),
    ]
    return ValidatorPipeline(validators)

def main():
    parser = argparse.ArgumentParser(description="Validation framework for context-dependent QA dataset.")
    parser.add_argument("--input", required=True, help="Путь к входному JSONL (проектные записи).")
    parser.add_argument("--out", required=True, help="Файл-вывод JSONL с результатами.")
    parser.add_argument("--summary", required=True, help="Файл сводных метрик (JSON).")
    parser.add_argument("--use-embeddings", action="store_true", help="Включить семантическую проверку ортогональности (OpenAI embeddings).")
    parser.add_argument("--max-sim", type=float, default=0.80, help="Порог косинусной близости контекстов (>= считается слишком похожим).")
    parser.add_argument("--max-contexts", type=int, default=8, help="Максимум контекстов для эвристики конечности.")
    parser.add_argument("--max-answer-tokens", type=int, default=2, help="Лимит слов/чисел в ответах и базовом ответе.")
    args = parser.parse_args()

    raw_items = load_jsonl(args.input)
    items: List[QAItem] = []
    for idx, r in enumerate(raw_items):
        try:
            items.append(QAItem.from_raw(r))
        except Exception as e:
            print(f"[WARN] Skip item #{idx}: {e}", file=sys.stderr)

    if not items:
        print("[ERROR] No parsable QA items. Check input file.", file=sys.stderr)
        sys.exit(1)
    items: List[QAItem] = [QAItem.from_raw(r) for r in raw_items]

    pipeline = build_pipeline(
        use_embeddings=args.use_embeddings,
        max_sim=args.max_sim,
        max_contexts=args.max_contexts,
        max_answer_tokens=args.max_answer_tokens,
    )

    results: List[ValidationResult] = []
    for it in items:
        res = pipeline.validate_item(it)
        results.append(res)

    # Сохраняем построчно детальные результаты
    out_rows: List[Dict] = []
    for r in results:
        out_rows.append({
            "f": r.item.f,
            "s": r.item.s,
            "j": r.item.j,
            "q": r.item.q,
            "n": r.item.n,
            "v": [{"c": vv.c, "a": vv.a} for vv in r.item.v],
            "passed": r.passed,
            "issues": [dataclasses.asdict(iss) for iss in r.issues],
            "metrics": r.metrics,
            "recommendations": r.recommendations,
            "ts": dt.datetime.now().isoformat()
        })
    write_jsonl(args.out, out_rows)

    # Сводка
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if any(i.severity == "error" for i in r.issues))
    warns = sum(1 for r in results if any(i.severity == "warn" for i in r.issues))
    avg_contexts = statistics.mean((len(r.item.v) for r in results), default=0.0)

    # Частоты типов проблем
    issue_counter = collections.Counter()
    for r in results:
        for i in r.issues:
            issue_counter[i.code] += 1
    top_issues = issue_counter.most_common(20)

    summary = {
        "total_items": total,
        "passed_items": passed,
        "failed_items": errors,
        "warn_items": warns,
        "pass_rate": (passed / total) if total else 0.0,
        "avg_contexts_per_item": avg_contexts,
        "top_issues": [{"code": c, "count": n} for c, n in top_issues],
        "params": {
            "use_embeddings": args.use_embeddings,
            "max_sim": args.max_sim,
            "max_contexts": args.max_contexts,
            "max_answer_tokens": args.max_answer_tokens,
        },
        "generated_at": dt.datetime.now().isoformat()
    }
    write_summary(args.summary, summary)

    print(f"[OK] Validated {total} items. Pass rate: {summary['pass_rate']:.2%}. Results -> {args.out}; Summary -> {args.summary}")

if __name__ == "__main__":
    main()
