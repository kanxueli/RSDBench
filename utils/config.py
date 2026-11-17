"""YAML config loading with deep merge and simple ${var} interpolation."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import yaml

_VAR = re.compile(r"\$\{([^}]+)\}")

# Portable path defaults (override with env; do not hardcode machine homes).
ENV_DATA_ROOT = "RSDBENCH_DATA_ROOT"
ENV_CACHE_ROOT = "RSDBENCH_CACHE_ROOT"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> str:
    """RSDBENCH_DATA_ROOT (or legacy RSDLIB_DATA_ROOT), else <repo_parent>/DATASET."""
    env = (
        os.environ.get(ENV_DATA_ROOT, "").strip()
        or os.environ.get("RSDLIB_DATA_ROOT", "").strip()
    )
    if env:
        return env
    return str((repo_root().parent / "DATASET").resolve())


def default_cache_root() -> str:
    """RSDBENCH_CACHE_ROOT (or legacy RSDLIB_CACHE_ROOT), else <repo>/.cache."""
    env = (
        os.environ.get(ENV_CACHE_ROOT, "").strip()
        or os.environ.get("RSDLIB_CACHE_ROOT", "").strip()
    )
    if env:
        return env
    return str((repo_root() / ".cache").resolve())


def apply_path_env_overrides(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Apply RSDBENCH_* (or legacy RSDLIB_*) env overrides; fill null roots."""
    env_data = os.environ.get(ENV_DATA_ROOT, "").strip() or os.environ.get("RSDLIB_DATA_ROOT", "").strip()
    env_cache = os.environ.get(ENV_CACHE_ROOT, "").strip() or os.environ.get("RSDLIB_CACHE_ROOT", "").strip()
    if env_data:
        cfg["data_root"] = env_data
    elif cfg.get("data_root") in (None, "", "null"):
        cfg["data_root"] = default_data_root()
    if env_cache:
        cfg["cache_root"] = env_cache
    elif cfg.get("cache_root") in (None, "", "null"):
        cfg["cache_root"] = default_cache_root()
    return cfg


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, Mapping):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _resolve_vars(obj: Any, root: Mapping[str, Any], stack: Optional[list] = None) -> Any:
    stack = stack or []
    if isinstance(obj, dict):
        return {k: _resolve_vars(v, root, stack) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_vars(v, root, stack) for v in obj]
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            key = m.group(1)
            if key in stack:
                raise ValueError(f"Circular config reference: {key}")
            parts = key.split(".")
            cur: Any = root
            for p in parts:
                if not isinstance(cur, Mapping) or p not in cur:
                    raise KeyError(f"Unknown config variable: ${{{key}}}")
                cur = cur[p]
            resolved = _resolve_vars(cur, root, stack + [key])
            return str(resolved)

        # full-string match keeps non-str types if single var
        full = _VAR.fullmatch(obj)
        if full:
            key = full.group(1)
            parts = key.split(".")
            cur: Any = root
            for p in parts:
                if not isinstance(cur, Mapping) or p not in cur:
                    raise KeyError(f"Unknown config variable: ${{{key}}}")
                cur = cur[p]
            return _resolve_vars(cur, root, stack + [key])
        return _VAR.sub(repl, obj)
    return obj


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def merge_configs(*cfgs: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cfg in cfgs:
        _deep_merge(out, cfg)
    return out


def load_experiment_config(
    model_cfg: Union[str, Path],
    protocol_cfg: Union[str, Path],
    data_cfg: Optional[Union[str, Path]] = None,
    train_cfg: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    base_model_cfg: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Merge protocol ← data ← train ← [base model] ← model/overlay ← CLI overrides."""
    root = Path(__file__).resolve().parents[1]
    pieces = [
        load_yaml(protocol_cfg),
        load_yaml(data_cfg or root / "configs" / "data" / "default.yaml"),
        load_yaml(train_cfg or root / "configs" / "train" / "default.yaml"),
    ]
    if base_model_cfg is not None:
        base_path = Path(base_model_cfg)
        overlay_path = Path(model_cfg).resolve()
        if base_path.exists() and base_path.resolve() != overlay_path:
            pieces.append(load_yaml(base_path))
    pieces.append(load_yaml(model_cfg))
    cfg = merge_configs(*pieces)
    if overrides:
        _deep_merge(cfg, overrides)
    # Env overrides before ${data_root}/... interpolation so dataset roots stay consistent.
    apply_path_env_overrides(cfg)
    return _resolve_vars(cfg, cfg)


def dump_resolved_config(cfg: Mapping[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, allow_unicode=True)


class ConfigDict(dict):
    """Dot-access wrapper used by models (TSlib-style configs object)."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(v, dict) and not isinstance(v, ConfigDict):
            v = ConfigDict(v)
            self[item] = v
        return v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for p in dotted.split("."):
            if not isinstance(cur, Mapping) or p not in cur:
                return default
            cur = cur[p]
        return cur
