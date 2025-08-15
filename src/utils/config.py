# src/utils/config.py
from __future__ import annotations
from pathlib import Path
import yaml
import copy
import datetime as dt

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_config(root_cfg_path: str | Path, override_paths: list[str | Path] | None = None) -> dict:
    root_cfg_path = Path(root_cfg_path)
    with root_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for p in override_paths or []:
        p = Path(p)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                _deep_update(cfg, yaml.safe_load(f))
        else:
            raise FileNotFoundError(f"Override file not found: {p}")

    # runtime 보조 필드 생성
    now = dt.datetime.now()
    namefmt = cfg["output"].get("namefmt", "%Y%m%d_%H%M%S")
    runstamp = now.strftime(namefmt)
    cfg["_runtime"] = {
        "runstamp": runstamp,
        "outdir": str(Path(cfg["output"]["dir"]) / cfg["eval"].get("name", "default") / runstamp)
    }
    return cfg

def pretty(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
