# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional, Type, TypeVar
from dynaconf import Dynaconf
import os, toml

# Файлы настроек: публичный и секретный
DEFAULT_SETTINGS_FILES = ("settings.conf", ".secrets.toml")

T = TypeVar("T")


def load_settings(paths: list[str] | None = None) -> Dict[str, Any]:
    """
    Грузим toml-файлы по очереди (поздние перекрывают ранние).
    """
    data: Dict[str, Any] = {}
    for p in paths or []:
        if p and os.path.exists(p):
            try:
                d = toml.load(p)

                # грубое поверхностное слияние
                def merge(a: dict, b: dict):
                    for k, v in b.items():
                        if isinstance(v, dict) and isinstance(a.get(k), dict):
                            merge(a[k], v)
                        else:
                            a[k] = v

                merge(data.setdefault("_root_", {}), d)  # сохраняем оригинал
                merge(data, d)
            except Exception:
                pass
    return data


def get_typed(dct: Dict[str, Any], dotted: str, default: T, typ: Type[T]) -> T:
    cur: Any = dct
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    try:
        return typ(cur)  # type: ignore
    except Exception:
        return default


def get_exp(settings: Dict[str, Any], exp_key: Optional[str]) -> Dict[str, Any]:
    """
    Возвращает словарь параметров эксперимента из секции [exp.<exp_key>].
    Если exp_key=None или секции нет — пустой словарь.
    """
    if not exp_key:
        return {}
    node = settings.get("exp", {})
    exp = node.get(exp_key, {})
    return exp if isinstance(exp, dict) else {}
