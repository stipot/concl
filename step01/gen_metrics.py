#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор отчёта по метрикам качества ГЕНЕРАЦИИ карточек (без оценки "решателя").
Поддерживает несколько входных jsonl для сравнения (например, gpt-4o и gpt-4o-mini).

Пример:
  python tools/gen_metrics.py \
      --in step01/data/gpt-4o_generated_questions_en.jsonl \
      --in step01/data/gpt-4o-mini_generated_questions_en.jsonl \
      --in-inval step01/data/gpt-4o_generated_questions_en_inval.jsonl \
      --out reports/gen_metrics_gpt.json

Автор: ты :)
"""
import argparse, json, os, sys, math, collections
from itertools import combinations
from typing import Dict, List, Tuple

WORD_RE = __import__("re").compile(r"\w+", __import__("re").U | __import__("re").M)


def read_jsonl(path: str) -> List[Dict]:
    items = []
    if not path or not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def tokenize_words(s: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer((s or "").strip())]


def normalize_answer(a: str) -> str:
    import re

    x = (a or "").strip()
    x = re.sub(r"^(?:the|a|an)\s+", "", x, flags=re.I)
    return x.lower()


def struct_pass(x: Dict) -> bool:
    if not isinstance(x, dict):
        return False
    if not isinstance(x.get("q", ""), str) or len(x.get("q", "").strip()) < 3:
        return False
    if not isinstance(x.get("n", ""), str) or len(x.get("n", "").strip()) < 1:
        return False
    v = x.get("v")
    if not isinstance(v, list) or not (2 <= len(v) <= 4):
        return False
    for vv in v:
        if not isinstance(vv, dict):
            return False
        if not isinstance(vv.get("c", ""), str) or len(vv.get("c", "").strip()) < 3:
            return False
        if not isinstance(vv.get("a", ""), str) or len(vv.get("a", "").strip()) < 1:
            return False
    return True


def quick_validate(x: Dict) -> Tuple[bool, List[str]]:
    errs = []
    q = (x.get("q") or "").strip()
    v = x.get("v") or []
    n = x.get("n", None)

    if not q:
        errs.append("q.missing")
    if not isinstance(v, list) or not (2 <= len(v) <= 4):
        errs.append("v.size")

    ans_norm = []
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
            # leakage
            if a and c and a.lower() in c.lower():
                errs.append(f"v[{i}].leak")
            ans_norm.append(normalize_answer(a))

    if ans_norm and len(set(ans_norm)) != len(ans_norm):
        errs.append("v.answers.dup")

    if n is None or str(n).strip() == "":
        errs.append("n.missing")
    else:
        n_norm = normalize_answer(str(n))
        if ans_norm and n_norm not in ans_norm:
            errs.append("n.not_aligned")

    return (len(errs) == 0), errs


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize_words(a.lower())), set(tokenize_words(b.lower()))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def orthogonality_ok(v: List[Dict], jaccard_max: float = 0.6) -> Tuple[bool, List[Tuple[int, int, float]]]:
    """
    Эвристика ортогональности: все пары контекстов имеют Жаккар < порога.
    Возвращаем флаг и список нарушений (i,j,score).
    """
    violations = []
    for (i, vi), (j, vj) in combinations(enumerate(v), 2):
        sim = jaccard(vi.get("c", ""), vj.get("c", ""))
        if sim >= jaccard_max:
            violations.append((i, j, sim))
    return (len(violations) == 0), violations


def compute_metrics(items: List[Dict], name: str, inv_items: List[Dict] = None, jaccard_max: float = 0.6, weights=None) -> Dict:
    inv_items = inv_items or []
    N = len(items)
    res = dict(name=name, total=N, invalid_total=len(inv_items))

    # базовые счётчики
    struct_ok = sum(struct_pass(x) for x in items)
    cpr_ok, leaks, uniq_ok, da_ok, conc_viol, ortho_ok = 0, 0, 0, 0, 0, 0
    ortho_violations = 0
    by_errors = collections.Counter()

    for x in items:
        sp = struct_pass(x)
        ok, errs = quick_validate(x)
        if ok:
            cpr_ok += 1
        for e in errs:
            by_errors[e] += 1
            if e.endswith(".leak"):
                leaks += 1
            if e == "v.answers.dup":
                pass  # дубликаты учтём отдельно
            if e.endswith(".a.too_long"):
                conc_viol += 1
            if e == "n.not_aligned":
                pass

        # AU (уникальность ответов)
        ans_norm = [normalize_answer(vv.get("a", "")) for vv in (x.get("v") or []) if isinstance(vv, dict)]
        if ans_norm and len(set(ans_norm)) == len(ans_norm):
            uniq_ok += 1

        # DA (default alignment)
        n_norm = normalize_answer(x.get("n", ""))
        if ans_norm and n_norm in ans_norm and ans_norm.count(n_norm) == 1:
            da_ok += 1

        # OH (ортогональность)
        ok_ortho, viol = orthogonality_ok(x.get("v") or [], jaccard_max=jaccard_max)
        if ok_ortho:
            ortho_ok += 1
        else:
            ortho_violations += len(viol)

    # нормированные метрики на карточку
    res.update(
        {
            "SPR": struct_ok / N if N else 0.0,
            "CPR": cpr_ok / N if N else 0.0,
            "LeakageRate": leaks / max(1, sum(len(x.get("v") or []) for x in items)),
            "AnswerUniqRate": uniq_ok / N if N else 0.0,
            "DefaultAlignRate": da_ok / N if N else 0.0,
            "ConcisionViolRate": conc_viol / max(1, sum(len(x.get("v") or []) for x in items)),
            "OrthoPassRate": ortho_ok / N if N else 0.0,
            "OrthoViolations": ortho_violations,
            "ErrorHistogram": dict(by_errors),
        }
    )

    # простая компоновка GenQuality (веса можно переопределить)
    weights = weights or dict(SPR=0.15, CPR=0.35, AnswerUniqRate=0.10, DefaultAlignRate=0.20, OrthoPassRate=0.15, LeakagePenalty=0.03, ConcisionPenalty=0.02)
    # штрафуем утечки и неконсиcтность
    gen_quality = (
        weights["SPR"] * res["SPR"]
        + weights["CPR"] * res["CPR"]
        + weights["AnswerUniqRate"] * res["AnswerUniqRate"]
        + weights["DefaultAlignRate"] * res["DefaultAlignRate"]
        + weights["OrthoPassRate"] * res["OrthoPassRate"]
        - weights["LeakagePenalty"] * res["LeakageRate"]
        - weights["ConcisionPenalty"] * res["ConcisionViolRate"]
    )
    res["GenQuality"] = max(0.0, min(1.0, gen_quality))

    # отчёт по invalid-файлу (если был)
    if inv_items:
        inv_errs = collections.Counter()
        for y in inv_items:
            for e in y.get("t") or []:
                inv_errs[e] += 1
        res["InvalidTopIssues"] = inv_errs.most_common(12)

    # разрезы по полям (если присутствуют)
    def grp(key):
        by = collections.Counter()
        for x in items:
            if key in x:
                by[str(x[key])] += 1
        return dict(by)

    for k in ("f", "s", "j"):
        res[f"dist_{k}"] = grp(k)

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="Путь(и) к jsonl валидных карточек.")
    ap.add_argument("--in-inval", dest="inputs_inval", action="append", default=[], help="Путь(и) к *_inval.jsonl.")
    ap.add_argument("--out", dest="out", required=True, help="Путь для json-отчёта.")
    ap.add_argument("--jaccard-max", type=float, default=0.6, help="Порог похожести контекстов для ортогональности.")
    args = ap.parse_args()

    # читаем все входы
    results = []
    inval_map = {}
    for p in args.inputs_inval:
        inval_map[os.path.basename(p)] = read_jsonl(p)

    for p in args.inputs:
        name = os.path.basename(p)
        items = read_jsonl(p)
        inv_guess = None
        # попытка сопоставить *_inval.jsonl по имени
        stem = name[:-6] if name.endswith(".jsonl") else name
        for b, inv in inval_map.items():
            if b.startswith(stem) or b.startswith(stem + "_inval"):
                inv_guess = inv
                break
        results.append(compute_metrics(items, name, inv_guess, jaccard_max=args.jaccard_max))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"reports": results}, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved → {os.path.abspath(args.out)}")
    # короткая сводка в stdout
    for r in results:
        print(
            f"- {r['name']}: N={r['total']}  GenQuality={r['GenQuality']:.3f}  SPR={r['SPR']:.3f}  CPR={r['CPR']:.3f}  DA={r['DefaultAlignRate']:.3f}  AU={r['AnswerUniqRate']:.3f}  LR={r['LeakageRate']:.3f}  ORTHO={r['OrthoPassRate']:.3f}"
        )


if __name__ == "__main__":
    main()
