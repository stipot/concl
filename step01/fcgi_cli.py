# fcgi_cli.py
# Unified "Finite-Context Generation Index" (FCGI) calculator for LLM question-generation quality.
# Usage examples:
#   python fcgi_cli.py compute --prefix ./step01/data/gpt-4o_validation \
#       --out-json ./step01/data/gpt-4o_fcgi.json --out-csv ./step01/data/gpt-4o_fcgi_by_subfield.csv
#
#   python fcgi_cli.py compute \
#       --valid ./step01/data/gpt-4o_validation_results.jsonl \
#       --inval ./step01/data/gpt-4o_validation_results_inval.jsonl \
#       --out-json ./step01/data/gpt-4o_fcgi.json
# python ./step01/fcgi_cli.py --prefix ./step01/data/gpt-4o_generated_questions_en/gpt-4o_validation --out-json ./step01/data/gpt-4o_generated_questions_en/gpt-4o_fcgi.json
# python ./step01/fcgi_cli.py --prefix ./step01/data/gpt-5-nano-2025-08-07_generated_questions_en/gpt-5-nano-2025-08-07_exp1_validation --out-json ./step01/data/gpt-5-nano-2025-08-07_generated_questions_en/gpt-gpt-5-nano-2025-08-07_exp1_fcgi.json
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

# --------- Defaults / constants ---------
TOKEN_RE = re.compile(r"\w+", flags=re.U | re.M)

DEFAULT_WEIGHTS = {
    "S": 2.0,  # structural soundness
    "A": 1.0,  # baseline alignment
    "U": 1.5,  # uniqueness (no answer collisions)
    "L": 2.0,  # leakage avoidance
    "O": 3.0,  # orthogonality
    "F": 1.0,  # finite diversity (no vague / too many)
    "C": 0.5,  # concision (answers length ≤ τ)
    "D": 0.75,  # default-arity proximity (k≈3)
}
DEFAULT_TAU = 7  # tokens threshold for C
DEFAULT_ALPHA = 0.75  # strength of global penalty by invalid_rate
DEFAULT_TARGET_K = 3.0  # desired average number of contexts

STRUCTURE_ERROR_HINTS = (
    ".missing",
    "contexts.too_few",
    "answer.empty",
    "context.empty",
)

KEY_CODES = [
    "baseline.not_aligned",
    "answers.duplicate",
    "leak.answer_in_context",
    "contexts.too_similar",
    "answer.vague",
    "contexts.too_many",
]


def extract_issues_and_v(record: dict):
    # issues: допускаем как список dict(code=...), так и строки
    issues = set()
    for it in record.get("issues", []):
        if isinstance(it, dict):
            code = it.get("code")
            if code:
                issues.add(code)
        elif isinstance(it, str):
            issues.add(it)

    # v_list: пытаемся найти на верхнем уровне, затем внутри {item,data,raw,original}
    v_list = record.get("v")
    if not isinstance(v_list, list):
        for key in ("item", "data", "raw", "original"):
            obj = record.get(key)
            if isinstance(obj, dict) and isinstance(obj.get("v"), list):
                v_list = obj["v"]
                break
    if not isinstance(v_list, list):
        v_list = []

    # санитизация [{c,a}]
    clean_v = []
    for vv in v_list:
        if isinstance(vv, dict):
            c = str(vv.get("c", "")).strip()
            a = str(vv.get("a", "")).strip()
            clean_v.append({"c": c, "a": a})
    return issues, clean_v


# --------- Data structures ---------
@dataclass
class Counts:
    total: int = 0
    structure_errors: int = 0
    code: Dict[str, int] = field(default_factory=lambda: Counter())
    answers_leq_tau_all: int = 0
    contexts_total_sum: int = 0  # for avg k

    def add_item(self, issues: Iterable[str], v_list: List[Dict], tau: int):
        self.total += 1
        # structure errors
        if any((h in c) for c in issues for h in STRUCTURE_ERROR_HINTS):
            self.structure_errors += 1
        # codes
        for c in KEY_CODES:
            if c in issues:
                self.code[c] += 1
        # concision C
        all_leq = True
        for vv in v_list:
            a = (vv.get("a") or "").strip()
            if len(TOKEN_RE.findall(a)) > tau:
                all_leq = False
                break
        if all_leq:
            self.answers_leq_tau_all += 1
        # k
        self.contexts_total_sum += len(v_list)

    def merge(self, other: "Counts"):
        self.total += other.total
        self.structure_errors += other.structure_errors
        self.answers_leq_tau_all += other.answers_leq_tau_all
        self.contexts_total_sum += other.contexts_total_sum
        self.code.update(other.code)


@dataclass
class Components:
    S: float
    A: float
    U: float
    L: float
    O: float
    F: float
    C: float
    D: float


# --------- IO helpers ---------
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                print(f"[WARN] Bad JSONL line skipped: {path}:{ln}", file=sys.stderr)


def write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------- Core computations ---------
def components_from_counts(cnt: Counts, weights: Dict[str, float], target_k: float) -> Tuple[Components, Dict[str, float]]:
    n = max(cnt.total, 1)

    S = 1.0 - (cnt.structure_errors / n)
    A = 1.0 - (cnt.code["baseline.not_aligned"] / n)
    U = 1.0 - (cnt.code["answers.duplicate"] / n)
    L = 1.0 - (cnt.code["leak.answer_in_context"] / n)
    O = 1.0 - (cnt.code["contexts.too_similar"] / n)
    F = 1.0 - ((cnt.code["answer.vague"] + cnt.code["contexts.too_many"]) / n)
    C = cnt.answers_leq_tau_all / n

    avg_k = cnt.contexts_total_sum / n
    # мягкий штраф вокруг target_k; превратим в [0,1]
    D = 1.0 - (abs(avg_k - target_k) ** 2)
    D = max(0.0, min(1.0, D))

    comps = Components(S=S, A=A, U=U, L=L, O=O, F=F, C=C, D=D)

    # лог-среднее с весами
    den = sum(weights.values())
    num = 0.0
    parts = {}
    for k in ["S", "A", "U", "L", "O", "F", "C", "D"]:
        v = getattr(comps, k)
        vv = max(v, 1e-9)
        num += weights[k] * math.log(vv)
        parts[k] = v

    fcgi_valid = math.exp(num / den)
    return comps, {"fcgi_valid": fcgi_valid, "avg_k": avg_k, **parts}


def aggregate_by_subfield(valid_path: str, tau: int) -> Dict[str, Counts]:
    agg: Dict[str, Counts] = defaultdict(Counts)
    for r in iter_jsonl(valid_path):
        s = str(r.get("s") or "").strip() or "(unknown)"
        issues, v_list = extract_issues_and_v(r)
        agg[s].add_item(issues, v_list, tau)
    return agg


def read_counts_from_results(valid_path: str, tau: int) -> Counts:
    cnt = Counts()
    for r in iter_jsonl(valid_path):
        issues, v_list = extract_issues_and_v(r)
        cnt.add_item(issues, v_list, tau)
    return cnt


def compute_invalid_rate(valid_total: int, inval_path: Optional[str]) -> Tuple[float, int]:
    if not inval_path or not os.path.exists(inval_path):
        return 0.0, 0
    inval_total = sum(1 for _ in iter_jsonl(inval_path))
    denom = max(valid_total + inval_total, 1)
    return inval_total / denom, inval_total


# --------- CLI ---------
@app.command("compute")
def cmd_compute(
    prefix: Optional[str] = typer.Option(None, help="Общий префикс файлов, будет добавлено _results.jsonl и _results_inval.jsonl"),
    valid: Optional[str] = typer.Option(None, help="Путь к {prefix}_validation_results.jsonl (если не используете --prefix)"),
    inval: Optional[str] = typer.Option(None, help="Путь к {prefix}_validation_results_inval.jsonl (опционально, для глобального штрафа)"),
    out_json: str = typer.Option(..., "--out-json", help="Путь для JSON отчёта (с компонентами, весами, разбивкой по subfield)"),
    out_csv: Optional[str] = typer.Option(None, "--out-csv", help="(Опционально) CSV-таблица метрик по subfield"),
    tau: int = typer.Option(DEFAULT_TAU, help="Порог токенов для компоненты C (≤ τ)"),
    alpha: float = typer.Option(DEFAULT_ALPHA, help="Коэффициент глобального штрафа по invalid_rate"),
    target_k: float = typer.Option(DEFAULT_TARGET_K, help="Желаемое среднее число контекстов (для компоненты D)"),
    # --weights-json '{"S":2,"A":1,"U":1.5,"L":2,"O":2,"F":1,"C":0.75,"D":0.25}'
    weights_json: Optional[str] = typer.Option(None, help="JSON-словарь весов компонентов (ключи S,A,U,L,O,F,C,D)"),
):
    """
    Считает FCGI по валидному корпусу и применяет глобальный штраф на долю inval.
    Сохраняет подробный JSON и (по желанию) CSV по subfield.
    """
    # resolve paths
    if prefix:
        valid_path = f"{prefix}_results.jsonl" if not prefix.endswith("_results") else f"{prefix}.jsonl"
        inval_path = f"{prefix}_results_inval.jsonl" if not prefix.endswith("_results") else f"{prefix}_inval.jsonl"
    else:
        if not valid:
            typer.secho("Нужно указать --prefix или --valid", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        valid_path = valid
        inval_path = inval

    if not os.path.exists(valid_path):
        typer.secho(f"Файл не найден: {valid_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # weights
    weights = dict(DEFAULT_WEIGHTS)
    if weights_json:
        try:
            user_w = json.loads(weights_json)
            for k, v in user_w.items():
                if k in weights:
                    weights[k] = float(v)
        except Exception as e:
            typer.secho(f"[WARN] Игнорирую некорректные weights_json: {e}", fg=typer.colors.YELLOW)

    # 1) Глобальные счета по валидным
    typer.secho(f"[INFO] Читаю валидные результаты: {valid_path}", fg=typer.colors.CYAN)
    cnt_valid = read_counts_from_results(valid_path, tau)

    # 2) Доля inval
    invalid_rate, inval_total = compute_invalid_rate(cnt_valid.total, inval_path)
    typer.secho(
        f"[INFO] Итого: valid={cnt_valid.total}, inval={inval_total}, invalid_rate={invalid_rate:.4f}",
        fg=typer.colors.BRIGHT_BLACK,
    )

    # 3) Компоненты и FCGI (валидная часть)
    comps, parts = components_from_counts(cnt_valid, weights, target_k)
    fcgi_valid = parts["fcgi_valid"]
    fcgi_global = fcgi_valid * (1.0 - alpha * invalid_rate)

    # 4) Разбивка по subfield
    by_s = aggregate_by_subfield(valid_path, tau)
    by_s_rows = []
    by_s_json = {}
    for s, cnt in by_s.items():
        c_s, p_s = components_from_counts(cnt, weights, target_k)
        fcgi_s = p_s["fcgi_valid"]
        row = {
            "subfield": s,
            "items": cnt.total,
            "fcgi_valid": round(fcgi_s, 6),
            "S": round(p_s["S"], 6),
            "A": round(p_s["A"], 6),
            "U": round(p_s["U"], 6),
            "L": round(p_s["L"], 6),
            "O": round(p_s["O"], 6),
            "F": round(p_s["F"], 6),
            "C": round(p_s["C"], 6),
            "D": round(p_s["D"], 6),
            "avg_k": round(p_s["avg_k"], 6),
            "structure_errors": cnt.structure_errors,
            "baseline.not_aligned": cnt.code["baseline.not_aligned"],
            "answers.duplicate": cnt.code["answers.duplicate"],
            "leak.answer_in_context": cnt.code["leak.answer_in_context"],
            "contexts.too_similar": cnt.code["contexts.too_similar"],
            "answer.vague": cnt.code["answer.vague"],
            "contexts.too_many": cnt.code["contexts.too_many"],
            "answers_leq_tau_all": cnt.answers_leq_tau_all,
        }
        by_s_rows.append(row)
        by_s_json[s] = {
            "counts": {
                "total": cnt.total,
                "structure_errors": cnt.structure_errors,
                "codes": dict(cnt.code),
                "answers_leq_tau_all": cnt.answers_leq_tau_all,
                "contexts_total_sum": cnt.contexts_total_sum,
            },
            "components": asdict(c_s),
            "parts": p_s,
        }

    # 5) Формируем JSON-отчёт
    report = {
        "meta": {
            "valid_path": os.path.abspath(valid_path),
            "inval_path": os.path.abspath(inval_path) if inval_path else None,
            "tau_tokens": tau,
            "alpha": alpha,
            "target_k": target_k,
            "weights": weights,
        },
        "totals": {
            "valid_items": cnt_valid.total,
            "inval_items": inval_total,
            "invalid_rate": invalid_rate,
        },
        "counts_valid": {
            "structure_errors": cnt_valid.structure_errors,
            "codes": dict(cnt_valid.code),
            "answers_leq_tau_all": cnt_valid.answers_leq_tau_all,
            "contexts_total_sum": cnt_valid.contexts_total_sum,
        },
        "components_valid": asdict(comps),
        "parts_valid": {k: (round(v, 12) if isinstance(v, float) else v) for k, v in parts.items()},
        "fcgi": {
            "fcgi_valid": round(fcgi_valid, 12),
            "fcgi_global": round(fcgi_global, 12),
        },
        "by_subfield": by_s_json,
    }

    write_json(out_json, report)
    typer.secho(f"[OK] JSON → {out_json}", fg=typer.colors.GREEN)

    if out_csv:
        # Сортируем по количеству записей, потом по FCGI
        by_s_rows.sort(key=lambda r: (-r["items"], -r["fcgi_valid"]))
        fieldnames = (
            list(by_s_rows[0].keys())
            if by_s_rows
            else ["subfield", "items", "fcgi_valid", "S", "A", "U", "L", "O", "F", "C", "D", "avg_k", "structure_errors", *KEY_CODES, "answers_leq_tau_all"]
        )
        write_csv(out_csv, by_s_rows, fieldnames)
        typer.secho(f"[OK] CSV → {out_csv}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
