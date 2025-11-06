# step01/rehab_inval.py
from __future__ import annotations
import json, os, re, sys, datetime as dt
from typing import List, Dict, Tuple
from openai import OpenAI
import toml

CRIT_LEAK = "leak.answer_in_context"
CRIT_DUP = "answers.duplicate"

PROMPT_FIX_LEAK = """System
You rewrite CONTEXTS to avoid leaking their own answers. Keep the paradigm change intact.

User
Task: For the given QA item, rewrite ONLY the 'c' fields that contain their own 'a' tokens. 
Rules:
- Do NOT alter q, n, or any 'a'.
- For each leaking context, replace explicit answer tokens/synonyms with neutral phrasing.
- Keep contexts ≤ 12 words; no hints of the exact answer.
- Return STRICT JSON in one of the two forms only:
A) {"v":[{"c":"..."}, ...]}  # same length and order as input v
OR
B) {"patch":{"<index>":{"c":"..."}, ...}}  # only for indices specified in "mask"
No extra keys, no prose.
"""

PROMPT_FIX_DUP = """System
You adjust only conflicting contexts to ensure answers across contexts are DISTINCT after normalization.

User
Task: For the given QA item, propose alternative paradigms/definitions for conflicting contexts so that each context yields a different short answer (≤10 tokens) already provided in 'a'.
Rules:
- Do NOT change 'q', 'n', or any 'a' values.
- Change only 'c' of conflicting indices.
- Keep contexts ≤ 12 words; mutually orthogonal paradigms.
Return STRICT JSON in one of the two forms only:
A) {"v":[{"c":"..."}, ...]}  # same length and order as input v
OR
B) {"patch":{"<index>":{"c":"..."}, ...}}  # only for indices specified in "mask"
No extra keys, no prose.
"""


def load_api_key():
    secrets = toml.load(".secrets.toml")
    return secrets.get("OPENAI_API_KEY")


def extract_leak_indices(issues: List[Dict]) -> List[int]:
    idx = []
    for iss in issues:
        if iss.get("code") == CRIT_LEAK and "v[" in (iss.get("detail", "") + iss.get("title", "")):
            # пытаемся вытащить индекс v[i] из текста (fallback)
            m = re.search(r"v\[(\d+)\]", iss.get("title", "") + iss.get("detail", ""))
            if m:
                idx.append(int(m.group(1)))
    return sorted(set(idx))


def has_duplicate_answers(v: List[Dict]) -> List[int]:
    # возвращаем индексы, которые входят в дубликатные группы
    norm = [re.sub(r"^(?:the|a|an)\s+", "", (x.get("a", "") + "").strip(), flags=re.I).lower() for x in v]
    out = []
    for i, a in enumerate(norm):
        if a and norm.count(a) > 1:
            out.append(i)
    return sorted(set(out))


def call_llm(client, model: str, system_user_prompt: str, q: str, n: str | None, v: List[Dict], mask: List[int] | None = None) -> Dict:
    # просим ДВА допустимых формата ответа: полный список v ИЛИ патч по индексам
    payload = {"q": q, "n": n, "v": v}
    if mask:
        payload["mask"] = mask  # подсказываем, какие индексы нужно менять
    msg = system_user_prompt + "\n\nInput JSON:\n" + json.dumps(payload, ensure_ascii=False)
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}], response_format={"type": "json_object"}, temperature=1.0, max_tokens=700)
    content = resp.choices[0].message.content
    data = json.loads(content)
    # нормализуем: всегда возвращаем словарь с ключами either {'v': [...]} OR {'patch': {'i': {'c': ...}}}
    if "v" in data and isinstance(data["v"], list):
        return {"v": data["v"]}
    if "patch" in data and isinstance(data["patch"], dict):
        return {"patch": data["patch"]}
    # fallback: если пришёл плоский список — трактуем как 'v'
    if isinstance(data, list):
        return {"v": [{"c": x, "a": None} if isinstance(x, str) else x for x in data]}
    return {}


def rehab_file(in_path: str, out_path: str, model="gpt-4o-mini"):
    key = load_api_key()
    client = OpenAI(api_key=key)
    ok, fixed, skipped = 0, 0, 0
    rows_out = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q, n, v = r["q"], r.get("n"), r["v"]
            issues = r.get("issues", [])
            # A) фиксим утечки
            leak_idx = extract_leak_indices(issues)
            if leak_idx:
                # помечаем те индексы, где надо переписать c (остальные c передадим как есть)
                v_in = [{"c": vv["c"], "a": vv["a"]} for vv in v]
                out = call_llm(client, model, PROMPT_FIX_LEAK, q, n, v_in, mask=leak_idx)
                # безопасно применяем
                if "v" in out and isinstance(out["v"], list):
                    # кейс: вернулся полный список той же длины
                    full = out["v"]
                    if len(full) == len(v):
                        for i in leak_idx:
                            if 0 <= i < len(full):
                                v[i]["c"] = full[i].get("c", v[i]["c"])
                    else:
                        # длины не совпали — не рискуем
                        print(f"[WARN] leak: length mismatch full={len(full)} vs v={len(v)}; skip full apply")
                if "patch" in out and isinstance(out["patch"], dict):
                    for k, patch in out["patch"].items():
                        try:
                            i = int(k)
                        except Exception:
                            continue
                        if 0 <= i < len(v) and (not leak_idx or i in leak_idx):
                            v[i]["c"] = patch.get("c", v[i]["c"])
                fixed += 1
            # B) фиксим дубликаты
            dup_idx = has_duplicate_answers(v)
            if dup_idx:
                v_in = [{"c": vv["c"], "a": vv["a"]} for vv in v]
                out = call_llm(client, model, PROMPT_FIX_DUP, q, n, v_in, mask=dup_idx)
                applied = False
                # формат 1: полный список "v" той же длины
                if "v" in out and isinstance(out["v"], list) and len(out["v"]) == len(v):
                    full = out["v"]
                    for i in dup_idx:
                        if 0 <= i < len(full):
                            v[i]["c"] = full[i].get("c", v[i]["c"])
                    applied = True
                # формат 2: патчи по индексам
                if "patch" in out and isinstance(out["patch"], dict):
                    for k, patch in out["patch"].items():
                        try:
                            i = int(k)
                        except Exception:
                            continue
                        if 0 <= i < len(v) and i in dup_idx:
                            v[i]["c"] = patch.get("c", v[i]["c"])
                    applied = True
                # формат 3: вернулся короткий список длиной ровно len(dup_idx) — трактуем как «по порядку»
                if (not applied) and "v" in out and isinstance(out["v"], list) and len(out["v"]) == len(dup_idx):
                    seq = out["v"]
                    for pos, i in enumerate(dup_idx):
                        if 0 <= i < len(v) and 0 <= pos < len(seq):
                            v[i]["c"] = (seq[pos].get("c") if isinstance(seq[pos], dict) else str(seq[pos])) or v[i]["c"]
                    applied = True
                if not applied:
                    print(f"[WARN] dup: cannot safely apply model output; keeping original contexts")
                fixed += 1

            ok += 1
            rows_out.append({**r, "v": v, "rehab_ts": dt.datetime.now().isoformat()})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for x in rows_out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[OK] processed={ok}, batches_fixed={fixed}, skipped={skipped}, out={out_path}")


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o-mini"
    rehab_file(in_path, out_path, model)
