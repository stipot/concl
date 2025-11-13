# genability_metric.py
from __future__ import annotations
import re, json, math, statistics, sqlite3, hashlib, os, threading
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Sequence
import numpy as np
import pandas as pd

# ========= NLTK (опционально) =========
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    _SMOOTH = SmoothingFunction().method1
    _HAS_NLTK = True

    def _sent_bleu(refs: list[list[Any]], hyp: list[Any]) -> float:
        refs_s = [_to_str_tokens(r) for r in (refs or []) if r]
        hyp_s = _to_str_tokens(hyp or [])
        if not refs_s or not hyp_s:
            return 0.0
        b = sentence_bleu(refs_s, hyp_s, smoothing_function=_SMOOTH)
        return float(b) if isinstance(b, (int, float)) else 0.0

except Exception:
    _HAS_NLTK = False
    _SMOOTH = None

    def _sent_bleu(refs: list[list[Any]], hyp: list[Any]) -> float:
        return 0.0


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _to_str_tokens(seq: Sequence[Any]) -> list[str]:
    return [str(t) for t in (seq or [])]


# ========= Токенизация + кеш =========
class Tokenizer:
    def __init__(self, name: str):
        self.name = name

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError


class RegexWordTokenizer(Tokenizer):
    def __init__(self):
        super().__init__("regex.word")

    def tokenize(self, text: str) -> list[str]:
        return _WORD_RE.findall((text or "").lower())


class TiktokenTokenizer(Tokenizer):
    def __init__(self, enc_name: str = "cl100k_base"):
        # ленивый импорт
        import tiktoken

        self.enc = tiktoken.get_encoding(enc_name)
        super().__init__(f"tiktoken.{enc_name}")

    def tokenize(self, text: str) -> list[str]:
        ids = self.enc.encode(text or "")
        return [f"▁{i}" for i in ids]


class HFTokenizer(Tokenizer):
    def __init__(self, pretrained: str):
        # ленивый импорт
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        super().__init__(f"hf.{pretrained}")

    def tokenize(self, text: str) -> list[str]:
        ids = self.tok.encode(text or "", add_special_tokens=False)
        return [f"▁{i}" for i in ids]


class TokenCache:
    _ddl = "CREATE TABLE IF NOT EXISTS tok_cache (k TEXT PRIMARY KEY, v TEXT NOT NULL);"

    def __init__(self, path: str = ".genability_tokcache.sqlite"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as db:
            db.execute(self._ddl)

    @staticmethod
    def _key(tok_name: str, text: str) -> str:
        h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        return f"{tok_name}:{h}"

    def get(self, tok_name: str, text: str) -> list[str] | None:
        k = self._key(tok_name, text)
        with self._lock, sqlite3.connect(self.path) as db:
            row = db.execute("SELECT v FROM tok_cache WHERE k=?", (k,)).fetchone()
            return None if not row else row[0].split("\x1f")

    def put(self, tok_name: str, text: str, tokens: list[str]) -> None:
        k = self._key(tok_name, text)
        v = "\x1f".join(tokens)
        with self._lock, sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO tok_cache(k,v) VALUES(?,?)", (k, v))


def make_tokenizer(pref: str | None = None) -> Tokenizer:
    if pref:
        kind, _, arg = pref.partition(":")
        try:
            if kind == "tiktoken":
                return TiktokenTokenizer(arg or "cl100k_base")
            if kind == "hf":
                return HFTokenizer(arg)
        except Exception:
            pass
    return RegexWordTokenizer()


_GLOBAL_TOK_CACHE = TokenCache()
_GLOBAL_TOKENIZER = make_tokenizer(os.getenv("GENABILITY_TOKENIZER"))


def tok(text: str) -> list[str]:
    cached = _GLOBAL_TOK_CACHE.get(_GLOBAL_TOKENIZER.name, text or "")
    if cached is not None:
        return cached
    toks = _GLOBAL_TOKENIZER.tokenize(text or "")
    _GLOBAL_TOK_CACHE.put(_GLOBAL_TOKENIZER.name, text or "", toks)
    return toks


# ========= утилиты метрик =========
def distinct_n(tokens: List[str], n: int = 2) -> float:
    if n <= 0 or len(tokens) < n:
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
    s = (s or "").strip().lower()
    s = re.sub(r"\b(a|an|the)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ========= проверка R1–R4 =========
def check_rule_block(items: List[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(items, list) or not items:
        return {"r1": 0.0, "r2": 0.0, "r3": 0.0, "r4": 1.0}
    r1 = r2 = r3 = r4 = 0
    total = 0
    for it in items:
        total += 1
        v = it.get("v", [])
        n = it.get("n", "")
        ok1 = isinstance(v, list) and 2 <= len(v) <= 4
        r1 += 1 if ok1 else 0
        v_norms: list[str] = []
        try:
            n_norm = normalize_answer(n)
            if isinstance(v, list):
                v_norms = [normalize_answer(str(x.get("a", ""))) for x in v]
            ok2 = n_norm in v_norms
        except Exception:
            ok2 = False
        r2 += 1 if ok2 else 0
        uniq = len(set(v_norms)) == len(v_norms) if v_norms else False
        r3 += 1 if uniq else 0
        q = (it.get("q") or "").lower()
        leak_indicators = ["answer is in the prompt", "see context above", "as provided earlier"]
        ok4 = not any(tok in q for tok in leak_indicators)
        r4 += 1 if ok4 else 0
    denom = max(1, total)
    return {"r1": r1 / denom, "r2": r2 / denom, "r3": r3 / denom, "r4": r4 / denom}


# ========= конфиг/веса =========
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
    max_prompt_overlap: float = 0.6
    diversity_n: int = 2
    min_entropy: float = 2.0
    field_item_keys: Tuple[str, ...] = ("q", "n", "v")
    tokenizer_pref: Optional[str] = None
    tokcache_path: str = ".genability_tokcache.sqlite"


def score_geometric(parts: Dict[str, float], weights: Weights) -> float:
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
    global _GLOBAL_TOKENIZER, _GLOBAL_TOK_CACHE
    if cfg.tokenizer_pref:
        _GLOBAL_TOKENIZER = make_tokenizer(cfg.tokenizer_pref)
    # переназначим кеш, если путь в конфиге иной
    if _GLOBAL_TOK_CACHE.path != cfg.tokcache_path:
        _GLOBAL_TOK_CACHE = TokenCache(cfg.tokcache_path)

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

        top_ok = False
        items = []
        item_struct_ok = 0.0
        if json_ok and isinstance(parsed, dict):
            top_ok = all(k in parsed for k in cfg.required_top_keys)
            items = parsed.get("items", [])
            good = 0
            for it in items if isinstance(items, list) else []:
                if all(key in it for key in cfg.field_item_keys):
                    good += 1
            item_struct_ok = good / max(1, len(items)) if isinstance(items, list) else 0.0

        schema_valid = 1.0 if (not expected_json) else (1.0 if (json_ok and top_ok) else 0.0)
        schema_valid = 0.5 * schema_valid + 0.5 * item_struct_ok

        rb = check_rule_block(items) if items else {"r1": 0.0, "r2": 0.0, "r3": 0.0, "r4": 1.0}
        rule_block = (rb["r1"] + rb["r2"] + rb["r3"] + rb["r4"]) / 4.0

        out = text
        if not out and items:
            parts = []
            for it in items:
                parts.append(str(it.get("q", "")))
                parts.append(str(it.get("n", "")))
                for vv in it.get("v", []):
                    parts.append(str(vv.get("a", "")))
            out = " ".join(parts)
        toks = tok(out or "")
        diversity = distinct_n(toks, n=cfg.diversity_n)

        prompt_toks = set(tok(r.get("prompt", "")))
        ans_toks = set(toks)
        overlap = jaccard(prompt_toks, ans_toks)
        overlap_score = 1.0 - min(1.0, overlap / cfg.max_prompt_overlap)
        entropy_score = min(1.0, char_entropy(out or "") / max(cfg.min_entropy, 1e-6))
        non_trivial = 0.6 * overlap_score + 0.4 * entropy_score

        rows.append(
            {
                "prompt_id": pid,
                "schema_valid": schema_valid,
                "rule_block": rule_block,
                "diversity": diversity,
                "non_trivial": non_trivial,
            }
        )

    # согласованность внутри prompt_id
    consistency_by_pid = {}
    for pid, lst in by_prompt.items():
        outs = []
        for r in lst:
            if r.get("output_text"):
                outs.append(r["output_text"])
            elif r.get("output_json"):
                ok, obj, _ = safe_json_parse(r["output_json"]) if isinstance(r["output_json"], str) else (True, r["output_json"], None)
                if ok and isinstance(obj, dict) and "items" in obj:
                    parts = []
                    for it in obj.get("items", []):
                        parts.append(str(it.get("n", "")))
                        for vv in it.get("v", []):
                            parts.append(str(vv.get("a", "")))
                    outs.append(" ".join(parts))
        if len(outs) <= 1 or not _HAS_NLTK:
            consistency_by_pid[pid] = 1.0
        else:
            tokenized = [tok(x) for x in outs]
            bleus = []
            for i, hyp in enumerate(tokenized):
                refs = tokenized[:i] + tokenized[i + 1 :]
                bleus.append(_sent_bleu(refs, hyp))
            mean_bleu = statistics.mean(bleus) if bleus else 0.0
            consistency_by_pid[pid] = 1.0 - min(1.0, mean_bleu)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["consistency"] = df["prompt_id"].map(consistency_by_pid).fillna(1.0)
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


# ========= CLI (Typer) =========
def _load_settings(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    import tomllib as toml  # Python 3.11+

    with open(path, "rb") as f:
        return toml.load(f)


def _records_from_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                recs.append(obj)
    return recs


def _write_reports(res: Dict[str, Any], out_dir: str, tag: str = "report") -> None:
    os.makedirs(out_dir, exist_ok=True)
    per_item: pd.DataFrame = res["per_item"]
    per_item.to_csv(os.path.join(out_dir, f"{tag}_per_item.csv"), index=False)
    with open(os.path.join(out_dir, f"{tag}_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(res["aggregate"], f, ensure_ascii=False, indent=2)


def _cfg_from_sources(settings: Dict[str, Any], **overrides) -> GenAbilityConfig:
    # извлекаем веса
    w = settings.get("weights", {})
    weights = Weights(
        schema_valid=float(w.get("schema_valid", overrides.get("w_schema_valid", 0.25))),
        rule_block=float(w.get("rule_block", overrides.get("w_rule_block", 0.25))),
        diversity=float(w.get("diversity", overrides.get("w_diversity", 0.20))),
        non_trivial=float(w.get("non_trivial", overrides.get("w_non_trivial", 0.15))),
        consistency=float(w.get("consistency", overrides.get("w_consistency", 0.15))),
    )
    g = settings.get("genability", {})
    return GenAbilityConfig(
        weights=weights,
        required_top_keys=tuple(g.get("required_top_keys", ("items",))),
        max_prompt_overlap=float(g.get("max_prompt_overlap", overrides.get("max_prompt_overlap", 0.6))),
        diversity_n=int(g.get("diversity_n", overrides.get("diversity_n", 2))),
        min_entropy=float(g.get("min_entropy", overrides.get("min_entropy", 2.0))),
        field_item_keys=tuple(g.get("field_item_keys", ("q", "n", "v"))),
        tokenizer_pref=str(overrides.get("tokenizer_pref", g.get("tokenizer_pref", None))) if (overrides.get("tokenizer_pref", None) or g.get("tokenizer_pref", None)) else None,
        tokcache_path=str(g.get("tokcache_path", overrides.get("tokcache_path", ".genability_tokcache.sqlite"))),
    )


# ---- Typer entrypoint
def _build_app():
    import typer

    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command("score")
    def score(
        inputs: List[str] = typer.Argument(..., help="Пути к JSONL с записями (records). Можно несколько."),
        out_dir: str = typer.Option("./reports", help="Куда писать отчёты"),
        settings_toml: Optional[str] = typer.Option(None, help="settings.toml с параметрами метрики"),
        tokenizer_pref: Optional[str] = typer.Option(None, help='Принудительный токенизатор: "regex" | "tiktoken:cl100k_base" | "hf:model"'),
        diversity_n: Optional[int] = typer.Option(None, help="n для distinct-n"),
        min_entropy: Optional[float] = typer.Option(None, help="Порог энтропии для нетривиальности"),
        max_prompt_overlap: Optional[float] = typer.Option(None, help="Нормирующий коэффициент для Jaccard"),
        w_schema_valid: Optional[float] = typer.Option(None, help="Вес компоненты schema_valid"),
        w_rule_block: Optional[float] = typer.Option(None, help="Вес компоненты rule_block"),
        w_diversity: Optional[float] = typer.Option(None, help="Вес компоненты diversity"),
        w_non_trivial: Optional[float] = typer.Option(None, help="Вес компоненты non_trivial"),
        w_consistency: Optional[float] = typer.Option(None, help="Вес компоненты consistency"),
        tag: str = typer.Option("report", help="Префикс имени файлов отчёта"),
    ):
        settings = _load_settings(settings_toml)
        cfg = _cfg_from_sources(
            settings,
            tokenizer_pref=tokenizer_pref,
            diversity_n=diversity_n,
            min_entropy=min_entropy,
            max_prompt_overlap=max_prompt_overlap,
            w_schema_valid=w_schema_valid,
            w_rule_block=w_rule_block,
            w_diversity=w_diversity,
            w_non_trivial=w_non_trivial,
            w_consistency=w_consistency,
        )
        recs = _records_from_jsonl(inputs)
        res = compute_metrics(recs, cfg=cfg)
        _write_reports(res, out_dir, tag)
        # короткий вывод в консоль
        print(json.dumps(res["aggregate"], ensure_ascii=False, indent=2))

    return app


# python genability_metric.py score ...
if __name__ == "__main__":
    try:
        import typer

        _build_app()()
    except ImportError:
        # если Typer не установлен, просто демонстрация
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
            {
                "prompt_id": "p1",
                "prompt": 'Generate 2–4 contexts; include baseline "n".',
                "output_text": "Q1 n=blue v:blue/red; Q2 n=cat v:dog/cat",
                "meta": {"expected_json": False},
            },
        ]
        res = compute_metrics(demo_records)
        print(res["aggregate"])
        print(res["per_item"].round(3))
