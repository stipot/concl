# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Iterable
from llm_provider import LLMProvider, LLMOptions


# ------- утилиты -------
def norm_ans(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I)
    return s.lower()


def load_generated_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # ожидаем поля q, n, v
            if isinstance(obj, dict) and obj.get("q") and obj.get("v"):
                items.append(obj)
    return items


# ------- построение запросов -------
def build_eval_prompt(q: str, ctx: Dict[str, str]) -> str:
    """
    Простой формат: просим КОРОТКИЙ ответ (<= 2 токенов), можно ужесточить JSONом.
    """
    c = ctx["c"]
    return f"""
You answer with ≤2 tokens. No explanation.
Question: {q}
Under this context: {c}
Return ONLY the final answer (≤2 tokens).
""".strip()


# ------- метрики -------
@dataclass
class EvalStats:
    total: int = 0
    baseline_acc: float = 0.0
    ctx_acc: float = 0.0
    sensitivity: float = 0.0  # доля случаев, где ответ меняется при смене парадигмы


def evaluate_model_on_dataset(
    provider: LLMProvider,
    model: str,
    data_path: str,
    limit: int = 200,
    use_responses: bool = True,
) -> Tuple[EvalStats, List[Dict[str, Any]]]:
    items = load_generated_items(data_path)[:limit]
    results: List[Dict[str, Any]] = []
    ok_baseline = 0
    ok_ctx = 0
    changed = 0

    for it in items:
        q = it["q"]
        n = it["n"]
        v = [x for x in it["v"] if isinstance(x, dict) and x.get("c") and x.get("a")]
        if not v:
            continue

        # baseline → выбираем тот контекст, где ответ совпадает с n (если есть)
        base_ctx = None
        for vv in v:
            if norm_ans(vv["a"]) == norm_ans(n):
                base_ctx = vv
                break
        if not base_ctx:
            base_ctx = v[0]  # fallback

        # другой контекст (другая парадигма)
        alt_ctx = None
        for vv in v:
            if norm_ans(vv["a"]) != norm_ans(base_ctx["a"]):
                alt_ctx = vv
                break
        if not alt_ctx:
            # все одинаковые ответы — пропустим элемент
            continue

        opt = LLMOptions(
            model=model,
            max_tokens=64,
            temperature=None,
            json_schema=None,
            json_mode="none",
            use_responses=use_responses,
            reasoning_effort="low" if use_responses else None,
            log_prompt=False,
            log_response=False,
        )

        # baseline ask
        p_base = build_eval_prompt(q, base_ctx)
        ans_base, _ = provider.generate_text(p_base, opt)
        ans_base = norm_ans(ans_base)

        # alt ask
        p_alt = build_eval_prompt(q, alt_ctx)
        ans_alt, _ = provider.generate_text(p_alt, opt)
        ans_alt = norm_ans(ans_alt)

        ok_baseline += int(ans_base == norm_ans(n))
        ok_ctx += int(ans_alt == norm_ans(alt_ctx["a"]))
        changed += int(ans_base != ans_alt)

        results.append(
            {
                "q": q,
                "baseline": {"ctx": base_ctx["c"], "gold": norm_ans(n), "pred": ans_base},
                "alt": {"ctx": alt_ctx["c"], "gold": norm_ans(alt_ctx["a"]), "pred": ans_alt},
                "changed": ans_base != ans_alt,
            }
        )

    total = max(1, len(results))
    stats = EvalStats(
        total=total,
        baseline_acc=ok_baseline / total,
        ctx_acc=ok_ctx / total,
        sensitivity=changed / total,
    )
    return stats, results
