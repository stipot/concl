# genability_metric.py
from __future__ import annotations
import re, json, math, statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd

# ------- утилиты токенизации / нормализации -------
try:
    # nltk нужен только для BLEU; если его нет, metric gracefully degrades
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    _SMOOTH = SmoothingFunction().method1
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False
    _SMOOTH = None

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def distinct_n(tokens: List[str], n: int = 2) -> float:
    if n <= 0:
        return 0.0
    if len(tokens) < n:
        return 0.0
    ngrams = set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return len(ngrams) / max(1, len(tokens) - n + 1)


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    from collections import Counter

    c = Counter(text)
    total = sum(c.values())
    probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def safe_json_parse(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        obj = json.loads(s)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def normalize_answer(s: str) -> str:
    # R3: lowercase, trim, remove articles
    s = (s or "").strip().lower()
    s = re.sub(r"\b(a|an|the)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ------- формат входа -------
"""
Ожидаемый вход:
records: List[Dict] где каждый элемент как минимум содержит:
{
  "prompt_id": "id-1",
  "prompt": "...",
  "output_text": "...",              # либо
  "output_json": "{...}",            # ожидаем наш формат с полями: items -> [ { "q": ..., "n": ..., "v": [ {"c": "...", "a": "..."} ... ] } ... ]
  "meta": { "expected_json": True, "required_keys": ["items"], ... }   # опционально
}

Если есть несколько генераций для одного prompt_id, модуль посчитает self-consistency.
"""


# ------- проверка правил R1-R4 для нашего формата -------
def check_rule_block(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Возвращает доли элементов, удовлетворяющих R1..R4:
      R1: 2–4 контекста в "v"
      R2: baseline n совпадает (семантически) с одним из ответов в "v"
      R3: ответы по контекстам различимы после нормализации
      R4: отсутствие явной утечки (признаки 'forbidden', 'leak', можно расширить)
    """
    if not isinstance(items, list) or not items:
        return {"r1": 0.0, "r2": 0.0, "r3": 0.0, "r4": 1.0}

    r1 = r2 = r3 = r4 = 0
    total = 0
    for it in items:
        total += 1
        v = it.get("v", [])
        n = it.get("n", "")
        # R1
        ok1 = isinstance(v, list) and 2 <= len(v) <= 4
        r1 += 1 if ok1 else 0
        # R2
        try:
            n_norm = normalize_answer(n)
            v_norms = [normalize_answer(str(x.get("a", ""))) for x in v]
            ok2 = n_norm in v_norms
        except Exception:
            ok2 = False
        r2 += 1 if ok2 else 0
        # R3
        try:
            # distinct после нормализации
            uniq = len(set(v_norms)) == len(v_norms) if v else False
        except Exception:
            uniq = False
        r3 += 1 if uniq else 0
        # R4 (грубая эвристика «нет утечки»)
        q = (it.get("q") or "").lower()
        leak_indicators = ["answer is in the prompt", "see context above", "as provided earlier"]
        ok4 = not any(tok in q for tok in leak_indicators)
        r4 += 1 if ok4 else 0

    denom = max(1, total)
    return {
        "r1": r1 / denom,
        "r2": r2 / denom,
        "r3": r3 / denom,
        "r4": r4 / denom,
    }


# ------- основная метрика -------
@dataclass
class Weights:
    schema_valid: float = 0.25
    rule_block: float = 0.25
    diversity: float = 0.20
    non_trivial: float = 0.15
    consistency: float = 0.15


@dataclass
class GenAbilityConfig:
    weights: Weights = Weights()
    required_top_keys: Tuple[str, ...] = ("items",)
    max_prompt_overlap: float = 0.6  # чем меньше перекрытие ответа с промптом, тем лучше
    diversity_n: int = 2  # distinct-n
    min_entropy: float = 2.0  # порог «нет воды»
    field_item_keys: Tuple[str, ...] = ("q", "n", "v")  # проверка структуры item


def score_geometric(parts: Dict[str, float], weights: Weights) -> float:
    # геометрическое среднее с весами; значения «клипуем» в [1e-6, 1]
    comps = []
    for key, w in [
        ("schema_valid", weights.schema_valid),
        ("rule_block", weights.rule_block),
        ("diversity", weights.diversity),
        ("non_trivial", weights.non_trivial),
        ("consistency", weights.consistency),
    ]:
        val = max(1e-6, min(1.0, parts.get(key, 0.0)))
        comps.append(val**w)
    return float(np.prod(comps))


def compute_metrics(records: List[Dict[str, Any]], cfg: GenAbilityConfig = GenAbilityConfig()) -> Dict[str, Any]:
    rows = []
    by_prompt: Dict[str, List[Dict]] = {}

    for r in records:
        pid = r.get("prompt_id") or f"__auto_{len(by_prompt)}"
        by_prompt.setdefault(pid, []).append(r)

        text = r.get("output_text")
        ojson_raw = r.get("output_json")
        expected_json = bool(r.get("meta", {}).get("expected_json", bool(ojson_raw)))
        json_ok, parsed, err = (False, None, None)
        if expected_json:
            if isinstance(ojson_raw, (dict, list)):
                parsed = ojson_raw
                json_ok = True
            elif isinstance(ojson_raw, str):
                json_ok, parsed, err = safe_json_parse(ojson_raw)

        # схема/структура
        top_ok = False
        items = []
        item_struct_ok = 0.0
        if json_ok and isinstance(parsed, dict):
            top_ok = all(k in parsed for k in cfg.required_top_keys)
            items = parsed.get("items", [])
            # проверим структуру каждого item
            good = 0
            for it in items if isinstance(items, list) else []:
                if all(key in it for key in cfg.field_item_keys):
                    good += 1
            item_struct_ok = good / max(1, len(items)) if isinstance(items, list) else 0.0

        schema_valid = 1.0 if (not expected_json) else (1.0 if (json_ok and top_ok) else 0.0)
        schema_valid = 0.5 * schema_valid + 0.5 * item_struct_ok

        # правила R1–R4 (только если есть items)
        rb = check_rule_block(items) if items else {"r1": 0.0, "r2": 0.0, "r3": 0.0, "r4": 1.0}
        rule_block = (rb["r1"] + rb["r2"] + rb["r3"] + rb["r4"]) / 4.0

        # разнообразие на уровне ответа (distinct-n)
        out = text
        if not out and items:
            # соберём текст из q/n/a для оценки различимости
            parts = []
            for it in items:
                parts.append(str(it.get("q", "")))
                parts.append(str(it.get("n", "")))
                for vv in it.get("v", []):
                    parts.append(str(vv.get("a", "")))
            out = " ".join(parts)
        toks = tokenize(out or "")
        diversity = distinct_n(toks, n=cfg.diversity_n)

        # нетривиальность: (1) низкое перекрытие с промптом; (2) достаточная «энтропия»
        prompt_toks = set(tokenize(r.get("prompt", "")))
        ans_toks = set(toks)
        overlap = jaccard(prompt_toks, ans_toks)  # 0..1
        overlap_score = 1.0 - min(1.0, overlap / cfg.max_prompt_overlap)  # >0 лучше
        entropy_score = min(1.0, char_entropy(out or "") / max(cfg.min_entropy, 1e-6))
        non_trivial = 0.6 * overlap_score + 0.4 * entropy_score

        # запишем построчно
        rows.append(
            {
                "prompt_id": pid,
                "schema_valid": schema_valid,
                "rule_block": rule_block,
                "diversity": diversity,
                "non_trivial": non_trivial,
            }
        )

    # согласованность по одному prompt_id (self-BLEU низкий -> хорошо)
    consistency_by_pid = {}
    for pid, lst in by_prompt.items():
        outs = []
        for r in lst:
            if r.get("output_text"):
                outs.append(r["output_text"])
            elif r.get("output_json"):
                ok, obj, _ = safe_json_parse(r["output_json"]) if isinstance(r["output_json"], str) else (True, r["output_json"], None)
                if ok and isinstance(obj, dict) and "items" in obj:
                    # склеим ответы
                    parts = []
                    for it in obj.get("items", []):
                        parts.append(str(it.get("n", "")))
                        for vv in it.get("v", []):
                            parts.append(str(vv.get("a", "")))
                    outs.append(" ".join(parts))
        if len(outs) <= 1 or not _HAS_NLTK:
            consistency_by_pid[pid] = 1.0  # нейтрально, если сравнивать нечего/нет nltk
        else:
            # self-BLEU -> diversity; затем нормируем
            bleus = []
            tokenized = [tokenize(x) for x in outs]
            for i, hyp in enumerate(tokenized):
                refs = tokenized[:i] + tokenized[i + 1 :]
                # BLEU с разглаживанием
                b = sentence_bleu(refs, hyp, smoothing_function=_SMOOTH)
                bleus.append(b)
            mean_bleu = statistics.mean(bleus) if bleus else 0.0
            consistency_by_pid[pid] = 1.0 - min(1.0, mean_bleu)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["consistency"] = df["prompt_id"].map(consistency_by_pid).fillna(1.0)
        # итог на запись
        df["GAI"] = df.apply(
            lambda r: score_geometric(
                {
                    "schema_valid": float(r["schema_valid"]),
                    "rule_block": float(r["rule_block"]),
                    "diversity": float(r["diversity"]),
                    "non_trivial": float(r["non_trivial"]),
                    "consistency": float(r["consistency"]),
                },
                cfg.weights,
            ),
            axis=1,
        )
    else:
        df = pd.DataFrame(columns=["prompt_id", "schema_valid", "rule_block", "diversity", "non_trivial", "consistency", "GAI"])

    # агрегаты
    agg = (
        {}
        if df.empty
        else {
            "schema_valid": df["schema_valid"].mean(),
            "rule_block": df["rule_block"].mean(),
            "diversity": df["diversity"].mean(),
            "non_trivial": df["non_trivial"].mean(),
            "consistency": df["consistency"].mean(),
            "GAI_mean": df["GAI"].mean(),
            "GAI_median": df["GAI"].median(),
        }
    )
    return {"per_item": df, "by_prompt_consistency": consistency_by_pid, "aggregate": agg, "config": cfg}


# ------- пример использования -------
if __name__ == "__main__":
    demo_records = [
        {
            "prompt_id": "p1",
            "prompt": 'Generate 2–4 contexts; include baseline "n".',
            "output_json": json.dumps(
                {
                    "items": [
                        {"q": "Q1", "n": "blue", "v": [{"c": "c1", "a": "blue"}, {"c": "c2", "a": "red"}]},
                        {"q": "Q2", "n": "cat", "v": [{"c": "c1", "a": "dog"}, {"c": "c2", "a": "cat"}]},
                    ]
                }
            ),
            "meta": {"expected_json": True},
        },
        {"prompt_id": "p1", "prompt": 'Generate 2–4 contexts; include baseline "n".', "output_text": "Q1 n=blue v:blue/red; Q2 n=cat v:dog/cat", "meta": {"expected_json": False}},
        {
            "prompt_id": "p2",
            "prompt": "Same rules.",
            "output_json": json.dumps({"items": [{"q": "Q3", "n": "a", "v": [{"c": "c1", "a": "a"}, {"c": "c2", "a": "a"}]}]}),
        },
    ]
    res = compute_metrics(demo_records)
    print(res["aggregate"])
    print(res["per_item"].round(3))
