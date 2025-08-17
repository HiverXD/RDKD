# src/utils/config.py
from __future__ import annotations
from pathlib import Path
import yaml
import copy
import datetime as dt
import os, json

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

# merge 이후 후처리 훅 추가
def finalize_softkd_paths(cfg):
    # 02_softkd일 때만 동작 (혹은 kd.enable이면 동작)
    if not cfg.get("kd", {}).get("enable", False):
        return cfg

    dname = cfg.get("data", {}).get("dataset")
    ckpt_root = cfg.get("paths", {}).get("ckpt_root", "src/model/ckpts")

    # 1) teacher ckpt: teachers.json > template > model.teacher.ckpt fallback
    # teachers.json(있으면)에서 dataset 키로 우선 탐색
    teachers_json = os.path.join(ckpt_root, "teachers.json")
    t_backbone, t_classifier = None, None
    if os.path.exists(teachers_json):
        j = json.load(open(teachers_json, "r", encoding="utf-8"))
        if dname in j:
            t_backbone = j[dname].get("backbone_path")
            t_classifier = j[dname].get("classifier_path")

    # 템플릿 대체
    t_backbone = t_backbone or cfg.get("paths", {}).get("template_backbone", "").format(dataset=dname)
    t_classifier = t_classifier or cfg.get("paths", {}).get("template_classifier", "").format(dataset=dname)

    cfg.setdefault("teacher", {})
    cfg["teacher"].setdefault("backbone_path", t_backbone)
    cfg["teacher"].setdefault("classifier_path", t_classifier)

    # 2) student ckpt (ImageNet 사전학습)
    cfg.setdefault("student", {})
    cfg["student"].setdefault("ckpt", os.path.join(ckpt_root, "ResNet18.pt"))

    # 3) logits cache
    cache_tmpl = cfg.get("paths", {}).get("template_cache", os.path.join(ckpt_root, "soft_targets/{dataset}_fp16.pt"))
    cfg.setdefault("kd", {})
    if cfg["kd"].get("cache_logits", True):
        cfg["kd"].setdefault("cache_path", cache_tmpl.format(dataset=dname))

    # 4) out_dir: 기존 규칙 유지 (runs/<experiment.name>/<dataset>)
    # train.py(또는 entry)에서 조합하므로 여기선 건드리지 않아도 됨.
    return cfg

def expand_softkd_templates(cfg):
    """dataset만으로 teacher ckpt, cache 경로 자동 확장"""
    ds = cfg["data"]["dataset"]
    # teacher ckpts
    t = cfg["model"]["teacher"]
    if "template_backbone" in t and "backbone_path" not in t:
        t["backbone_path"] = t["template_backbone"].format(dataset=ds)
    if "template_classifier" in t and "classifier_path" not in t:
        t["classifier_path"] = t["template_classifier"].format(dataset=ds)
    # cache logits
    kdcfg = cfg["kd"]
    if kdcfg.get("cache_logits", False):
        if "cache_template" in kdcfg and "cache_path" not in kdcfg:
            kdcfg["cache_path"] = kdcfg["cache_template"].format(dataset=ds)
    return cfg
